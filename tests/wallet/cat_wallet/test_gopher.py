from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest
from chia_rs import AugSchemeMPL, G2Element

from chia.clvm.spend_sim import CostLogger, SimClient, SpendSim, sim_and_client
from chia.consensus.default_constants import DEFAULT_CONSTANTS
from chia.types.announcement import Announcement
from chia.types.blockchain_format.coin import Coin
from chia.types.blockchain_format.program import Program
from chia.types.blockchain_format.sized_bytes import bytes32
from chia.types.coin_spend import make_spend
from chia.types.mempool_inclusion_status import MempoolInclusionStatus
from chia.types.spend_bundle import SpendBundle
from chia.util.errors import Err
from chia.util.ints import uint64
from chia.wallet.cat_wallet.cat_utils import (
    CAT_MOD,
    SpendableCAT,
    construct_cat_puzzle,
    unsigned_spend_bundle_for_spendable_cats,
)
from chia.wallet.cat_wallet.gopher_utils import GOPHER_TAIL_MOD, OFFER_MOD_HASH, get_gopher_tail_solution
from chia.wallet.lineage_proof import LineageProof
from chia.wallet.outer_puzzles import AssetType
from chia.wallet.payment import Payment
from chia.wallet.puzzle_drivers import PuzzleInfo
from chia.wallet.trading.offer import NotarizedPayment, Offer
from tests.clvm.benchmark_costs import cost_of_spend_bundle
from tests.conftest import ConsensusMode

acs = Program.to(1)
acs_ph = acs.get_tree_hash()
NO_LINEAGE_PROOF = LineageProof()


# Some methods mapping strings to CATs
def str_to_tail(tail_str: str) -> Program:
    tail: Program = Program.to([3, [], [1, tail_str], []])
    return tail


def str_to_tail_hash(tail_str: str) -> bytes32:
    tail_hash: bytes32 = Program.to([3, [], [1, tail_str], []]).get_tree_hash()
    return tail_hash


async def generate_coins(
    sim: SpendSim,
    sim_client: SimClient,
    requested_coins: Dict[Optional[str], List[uint64]],
) -> Dict[Optional[str], List[Coin]]:
    await sim.farm_block(acs_ph)
    parent_coin: Coin = [cr.coin for cr in await sim_client.get_coin_records_by_puzzle_hash(acs_ph)][0]

    # We need to gather a list of initial coins to create as well as spends that do the eve spend for every CAT
    payments: List[Payment] = []
    cat_bundles: List[SpendBundle] = []
    for tail_str, amounts in requested_coins.items():
        for amount in amounts:
            if tail_str:
                tail: Program = str_to_tail(tail_str)  # Making a fake but unique TAIL
                cat_puzzle: Program = construct_cat_puzzle(CAT_MOD, tail.get_tree_hash(), acs)
                payments.append(Payment(cat_puzzle.get_tree_hash(), amount))
                cat_bundles.append(
                    unsigned_spend_bundle_for_spendable_cats(
                        CAT_MOD,
                        [
                            SpendableCAT(
                                Coin(parent_coin.name(), cat_puzzle.get_tree_hash(), amount),
                                tail.get_tree_hash(),
                                acs,
                                Program.to([[51, acs_ph, amount], [51, 0, -113, tail, []]]),
                            )
                        ],
                    )
                )
            else:
                payments.append(Payment(acs_ph, amount))

    # This bundle creates all of the initial coins
    parent_bundle = SpendBundle(
        [
            make_spend(
                parent_coin,
                acs,
                Program.to([[51, p.puzzle_hash, p.amount] for p in payments]),
            )
        ],
        G2Element(),
    )

    # Then we aggregate it with all of the eve spends
    await sim_client.push_tx(SpendBundle.aggregate([parent_bundle, *cat_bundles]))
    await sim.farm_block()

    # Search for all of the coins and put them into a dictionary
    coin_dict: Dict[Optional[str], List[Coin]] = {}
    for tail_str, _ in requested_coins.items():
        if tail_str:
            tail_hash: bytes32 = str_to_tail_hash(tail_str)
            cat_ph: bytes32 = construct_cat_puzzle(CAT_MOD, tail_hash, acs).get_tree_hash()
            coin_dict[tail_str] = [
                cr.coin for cr in await sim_client.get_coin_records_by_puzzle_hash(cat_ph, include_spent_coins=False)
            ]
        else:
            coin_dict[None] = list(
                filter(
                    lambda c: c.amount < 250000000000,
                    [
                        cr.coin
                        for cr in await sim_client.get_coin_records_by_puzzle_hash(acs_ph, include_spent_coins=False)
                    ],
                )
            )

    return coin_dict


async def do_spend(
    sim: SpendSim,
    sim_client: SimClient,
    tail: Program,
    coins: List[Coin],
    lineage_proofs: List[Program],
    inner_solutions: List[Program],
    expected_result: Tuple[MempoolInclusionStatus, Err],
    reveal_limitations_program: bool = True,
    signatures: List[G2Element] = [],
    extra_deltas: Optional[List[int]] = None,
    additional_spends: List[SpendBundle] = [],
    limitations_solutions: Optional[List[Program]] = None,
    cost_logger: Optional[CostLogger] = None,
    cost_log_msg: str = "",
) -> int:
    if limitations_solutions is None:
        limitations_solutions = [Program.to([])] * len(coins)
    if extra_deltas is None:
        extra_deltas = [0] * len(coins)

    spendable_cat_list: List[SpendableCAT] = []
    for coin, innersol, proof, limitations_solution, extra_delta in zip(
        coins, inner_solutions, lineage_proofs, limitations_solutions, extra_deltas
    ):
        spendable_cat_list.append(
            SpendableCAT(
                coin,
                tail.get_tree_hash(),
                acs,
                innersol,
                limitations_solution=limitations_solution,
                lineage_proof=proof,
                extra_delta=extra_delta,
                limitations_program_reveal=tail if reveal_limitations_program else Program.to([]),
            )
        )

    spend_bundle: SpendBundle = unsigned_spend_bundle_for_spendable_cats(
        CAT_MOD,
        spendable_cat_list,
    )
    agg_sig = AugSchemeMPL.aggregate(signatures)
    final_bundle = SpendBundle.aggregate(
        [
            *additional_spends,
            spend_bundle,
            SpendBundle([], agg_sig),  # "Signing" the spend bundle
        ]
    )
    if cost_logger is not None:
        final_bundle = cost_logger.add_cost(cost_log_msg, final_bundle)
    result = await sim_client.push_tx(final_bundle)
    assert result == expected_result
    cost = cost_of_spend_bundle(spend_bundle)
    await sim.farm_block()
    return cost


def generate_secure_bundle(
    selected_coins: List[Coin],
    announcements: List[Announcement],
    offered_amount: uint64,
    tail_str: Optional[str] = None,
    inner_solution: List[List[Any]] = [],
) -> SpendBundle:
    announcement_assertions: List[List[Any]] = [[63, a.name()] for a in announcements]
    selected_coin_amount: int = sum([c.amount for c in selected_coins])
    non_primaries: List[Coin] = [] if len(selected_coins) < 2 else selected_coins[1:]
    if inner_solution:
        inner_solution += announcement_assertions
    else:
        inner_solution = [
            [51, Offer.ph(), offered_amount],
            [51, acs_ph, uint64(selected_coin_amount - offered_amount)],
            *announcement_assertions,
        ]

    if tail_str is None:
        bundle = SpendBundle(
            [
                make_spend(
                    selected_coins[0],
                    acs,
                    Program.to(inner_solution),
                ),
                *[make_spend(c, acs, Program.to([])) for c in non_primaries],
            ],
            G2Element(),
        )
    else:
        spendable_cats: List[SpendableCAT] = [
            SpendableCAT(
                c,
                str_to_tail_hash(tail_str),
                acs,
                Program.to(
                    [
                        [51, 0, -113, str_to_tail(tail_str), Program.to([])],  # Use the TAIL rather than lineage
                        *(inner_solution if c == selected_coins[0] else []),
                    ]
                ),
            )
            for c in selected_coins
        ]
        bundle = unsigned_spend_bundle_for_spendable_cats(CAT_MOD, spendable_cats)

    return bundle


class TestGopherLifecycle:
    @pytest.mark.limit_consensus_modes(allowed=[ConsensusMode.PLAIN, ConsensusMode.HARD_FORK_2_0], reason="save time")
    @pytest.mark.anyio
    async def test_gopher_tail(self, consensus_mode: ConsensusMode) -> None:
        async with sim_and_client() as (sim, sim_client):
            sim.pass_blocks(DEFAULT_CONSTANTS.HARD_FORK_HEIGHT)
            coins_needed: Dict[Optional[str], List[uint64]] = {
                None: [uint64(100000)],
                "red": [uint64(100000)],
            }
            cat_tail_hash = str_to_tail_hash("red")
            gopher_tail = GOPHER_TAIL_MOD.curry(OFFER_MOD_HASH)
            gopher_tail_hash = gopher_tail.get_tree_hash()

            all_coins: Dict[Optional[str], List[Coin]] = await generate_coins(sim, sim_client, coins_needed)
            xch_coin = all_coins[None][0]
            cat_coin = all_coins["red"][0]

            offered = uint64(100)  # xch
            asked = uint64(1000)  # cat
            mint_amount = 10
            gopher_puzzle_hash: bytes32 = construct_cat_puzzle(CAT_MOD, gopher_tail_hash, acs).get_tree_hash()

            # create an offer of xch for cat
            driver_dict: Dict[bytes32, PuzzleInfo] = {
                cat_tail_hash: PuzzleInfo({"type": AssetType.CAT.value, "tail": "0x" + cat_tail_hash.hex()}),
            }

            driver_dict_as_infos: Dict[str, Any] = {}
            for key, value in driver_dict.items():
                driver_dict_as_infos[key.hex()] = value.info

            # Create an XCH Offer for RED
            chia_requested_pmts: Dict[Optional[bytes32], List[Payment]] = {
                cat_tail_hash: [
                    Payment(acs_ph, asked, [b"memo"]),
                ]
            }

            chia_requested_payments: Dict[Optional[bytes32], List[NotarizedPayment]] = Offer.notarize_payments(
                chia_requested_pmts, [xch_coin]
            )
            chia_announcements: List[Announcement] = Offer.calculate_announcements(chia_requested_payments, driver_dict)
            inner_solution: List[List[Any]] = [
                [51, Offer.ph(), offered],
                [51, gopher_puzzle_hash, mint_amount],
                [51, acs_ph, uint64(xch_coin.amount - (offered + mint_amount))],
                [60, Offer.ph()],
            ]
            chia_secured_bundle: SpendBundle = generate_secure_bundle(
                [xch_coin], chia_announcements, offered, inner_solution=inner_solution
            )
            chia_offer = Offer(chia_requested_payments, chia_secured_bundle, driver_dict)
            assert not chia_offer.is_valid()

            # create an offer of cat for xch
            cat_requested_pmts: Dict[Optional[bytes32], List[Payment]] = {
                None: [
                    Payment(acs_ph, uint64(51), [b"cat memo"]),
                    Payment(acs_ph, uint64(49), [b"cat memo"]),
                ]
            }

            cat_requested_payments: Dict[Optional[bytes32], List[NotarizedPayment]] = Offer.notarize_payments(
                cat_requested_pmts, [cat_coin]
            )
            cat_announcements: List[Announcement] = Offer.calculate_announcements(cat_requested_payments, driver_dict)
            cat_secured_bundle: SpendBundle = generate_secure_bundle(
                [cat_coin], cat_announcements, asked, tail_str="red"
            )
            cat_offer = Offer(cat_requested_payments, cat_secured_bundle, driver_dict)
            assert not cat_offer.is_valid()

            new_offer = Offer.aggregate([chia_offer, cat_offer])
            assert new_offer.is_valid()
            offer_bundle: SpendBundle = new_offer.to_valid_spend()

            cat_mod_hash = CAT_MOD.get_tree_hash()
            cat_mod_hash_hash = Program.to(cat_mod_hash).get_tree_hash()
            eve_coin = list(filter(lambda a: a.amount == mint_amount, chia_secured_bundle.additions()))[0]
            truths = [
                [acs_ph, cat_mod_hash, cat_mod_hash_hash, gopher_tail_hash],
                eve_coin.name(),
                eve_coin.parent_coin_info,
                eve_coin.puzzle_hash,
                eve_coin.amount,
            ]
            inner_conds = [[51, acs_ph, mint_amount]]

            tail_sol = get_gopher_tail_solution(new_offer.requested_payments)
            full_tail_solution = Program.to([truths, 0, 0, 0, inner_conds, tail_sol])

            # test that the tail will run
            tail_output = gopher_tail.run(full_tail_solution)
            assert tail_output

            tail_condition = [51, 0, -113, gopher_tail, tail_sol]
            cat_inner_solution: Program = Program.to([tail_condition, [51, acs_ph, mint_amount]])
            mint_spend = unsigned_spend_bundle_for_spendable_cats(
                CAT_MOD,
                [
                    SpendableCAT(
                        list(filter(lambda a: a.amount == mint_amount, chia_secured_bundle.additions()))[0],
                        gopher_tail_hash,
                        acs,
                        cat_inner_solution,
                        limitations_program_reveal=gopher_tail,
                    )
                ],
            )

            full_spend = SpendBundle.aggregate([mint_spend, offer_bundle])
            result = await sim_client.push_tx(full_spend)
            assert result == (MempoolInclusionStatus.SUCCESS, None)
            await sim.farm_block()

            new_coin = (
                await sim_client.get_coin_records_by_puzzle_hash(gopher_puzzle_hash, include_spent_coins=False)
            )[0].coin
            assert new_coin.puzzle_hash == gopher_puzzle_hash
            assert new_coin.amount == mint_amount
