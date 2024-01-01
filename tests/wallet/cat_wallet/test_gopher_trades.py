from __future__ import annotations

from typing import Any, Dict, Union

import pytest

from chia.types.blockchain_format.sized_bytes import bytes32
from chia.util.ints import uint32, uint64
from chia.wallet.cat_wallet.cat_wallet import CATWallet
from chia.wallet.outer_puzzles import AssetType
from chia.wallet.puzzle_drivers import PuzzleInfo
from chia.wallet.trading.offer import Offer
from tests.conftest import SOFTFORK_HEIGHTS, ConsensusMode
from tests.environments.wallet import WalletEnvironment, WalletStateTransition, WalletTestFramework


@pytest.mark.limit_consensus_modes(allowed=[ConsensusMode.HARD_FORK_2_0], reason="save time")
@pytest.mark.anyio
@pytest.mark.parametrize(
    "wallet_environments,credential_restricted,active_softfork_height",
    [
        (
            {"num_environments": 2, "trusted": True, "blocks_needed": [1, 1], "reuse_puzhash": True},
            True,
            SOFTFORK_HEIGHTS[0],
        ),
    ],
    indirect=["wallet_environments"],
)
async def test_cat_trades(
    wallet_environments: WalletTestFramework,
    credential_restricted: bool,
    active_softfork_height: uint32,
) -> None:
    # Setup
    env_maker: WalletEnvironment = wallet_environments.environments[0]
    env_taker: WalletEnvironment = wallet_environments.environments[1]
    wallet_node_maker = env_maker.node
    wallet_node_taker = env_taker.node
    # client_maker = env_maker.rpc_client
    # client_taker = env_taker.rpc_client
    wallet_maker = env_maker.xch_wallet
    # wallet_taker = env_taker.xch_wallet
    # full_node = wallet_environments.full_node

    # trusted = len(wallet_node_maker.config["trusted_peers"]) > 0

    # Aliasing
    env_maker.wallet_aliases = {
        "xch": 1,
        "cat": 2,
    }
    env_taker.wallet_aliases = {
        "xch": 1,
        "cat": 3,
        "gopher": 2,
    }

    # Mint some standard CATs
    async with wallet_node_maker.wallet_state_manager.lock:
        cat_wallet_maker = await CATWallet.create_new_cat_wallet(
            wallet_node_maker.wallet_state_manager,
            wallet_maker,
            {"identifier": "genesis_by_id"},
            uint64(100),
            wallet_environments.tx_config,
        )

    await wallet_environments.process_pending_states(
        [
            # Balance checking for this scenario is covered in test_cat_wallet
            WalletStateTransition(
                pre_block_balance_updates={
                    "xch": {"set_remainder": True},
                    "cat": {"init": True, "set_remainder": True},
                },
                post_block_balance_updates={
                    "xch": {"set_remainder": True},
                    "cat": {"set_remainder": True},
                },
            ),
            WalletStateTransition(
                pre_block_balance_updates={
                    "xch": {"set_remainder": True},
                },
                post_block_balance_updates={
                    "xch": {"set_remainder": True},
                },
            ),
        ]
    )

    OfferSummary = Dict[Union[int, bytes32], int]
    cat_for_chia: OfferSummary = {
        wallet_maker.id(): 100,
        cat_wallet_maker.id(): -10,  # The taker has no knowledge of this CAT yet
    }
    driver_dict: Dict[bytes32, PuzzleInfo] = {}

    asset_id: str = cat_wallet_maker.get_asset_id()
    driver_item: Dict[str, Any] = {
        "type": AssetType.CAT.value,
        "tail": "0x" + asset_id,
    }
    driver_dict[bytes32.from_hexstr(asset_id)] = PuzzleInfo(driver_item)

    trade_manager_maker = env_maker.wallet_state_manager.trade_manager
    trade_manager_taker = env_taker.wallet_state_manager.trade_manager

    # chia_for_cat
    success, trade_make, error = await trade_manager_maker.create_offer_for_ids(
        cat_for_chia, wallet_environments.tx_config
    )
    assert error is None
    assert success is True
    assert trade_make is not None

    peer = wallet_node_taker.get_full_node_peer()
    trade_take, tx_records = await trade_manager_taker.respond_to_offer(
        Offer.from_bytes(trade_make.offer),
        peer,
        wallet_environments.tx_config,
        mint_gophers=True,
    )
    assert trade_take is not None
    assert tx_records is not None

    await wallet_environments.process_pending_states(
        [
            WalletStateTransition(
                pre_block_balance_updates={
                    "xch": {
                        "set_remainder": True,
                    },
                    "cat": {
                        "pending_coin_removal_count": 1,
                        "spendable_balance": -100,
                        "max_send_amount": -100,
                    },
                },
                post_block_balance_updates={
                    "xch": {
                        "pending_coin_removal_count": 0,
                        "confirmed_wallet_balance": 100,
                        "spendable_balance": 100,
                        "max_send_amount": 100,
                        "unconfirmed_wallet_balance": 100,
                        "unspent_coin_count": 1,
                    },
                    "cat": {"confirmed_wallet_balance": -10, "unconfirmed_wallet_balance": -10, "set_remainder": True},
                },
            ),
            WalletStateTransition(
                pre_block_balance_updates={
                    "xch": {
                        "set_remainder": True,
                    },
                    "cat": {
                        "init": True,
                        "set_remainder": True,
                    },
                    "gopher": {
                        "init": True,
                        "set_remainder": True,
                    },
                },
                post_block_balance_updates={
                    "xch": {"confirmed_wallet_balance": -110, "set_remainder": True},
                    "cat": {"confirmed_wallet_balance": 10, "set_remainder": True},
                    "gopher": {"confirmed_wallet_balance": 10, "set_remainder": True},
                },
            ),
        ],
    )
