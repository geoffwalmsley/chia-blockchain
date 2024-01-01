from __future__ import annotations

from typing import Any, Dict, List, Optional

from chia.types.blockchain_format.program import Program
from chia.types.blockchain_format.sized_bytes import bytes32
from chia.util.ints import uint64
from chia.wallet.cat_wallet.cat_utils import CAT_MOD
from chia.wallet.outer_puzzles import AssetType
from chia.wallet.puzzle_drivers import PuzzleInfo
from chia.wallet.puzzles.load_clvm import load_clvm_maybe_recompile
from chia.wallet.trading.offer import NotarizedPayment
from chia.wallet.util.curry_and_treehash import calculate_hash_of_quoted_mod_hash, curry_and_treehash

CAT_MOD_HASH = CAT_MOD.get_tree_hash()
GOPHER_TAIL_MOD = load_clvm_maybe_recompile("gopher_tail.clsp", package_or_requirement="chia.wallet.cat_wallet.puzzles")
OFFER_MOD = load_clvm_maybe_recompile("settlement_payments.clsp", package_or_requirement="chia.wallet.puzzles")
OFFER_MOD_HASH = OFFER_MOD.get_tree_hash()
GOPHER_TAIL_PUZZLE = GOPHER_TAIL_MOD.curry(OFFER_MOD_HASH)
GOPHER_TAIL_PUZZLE_HASH = GOPHER_TAIL_PUZZLE.get_tree_hash()


def get_gopher_puzzle_hash(inner_puzzle_hash: bytes32) -> bytes32:
    mod_hash_hash = calculate_hash_of_quoted_mod_hash(CAT_MOD_HASH)

    hashed_arguments = [
        Program.to(CAT_MOD_HASH).get_tree_hash(),
        Program.to(GOPHER_TAIL_PUZZLE_HASH).get_tree_hash(),
        inner_puzzle_hash,
    ]
    return curry_and_treehash(mod_hash_hash, *hashed_arguments)


def get_gopher_tail_solution(requested_payments: Dict[Optional[bytes32], List[Any]]) -> Program:
    asset_pmts = []
    for asset_id, payments in requested_payments.items():
        assert isinstance(payments[0], NotarizedPayment)
        pmt = Program.to((payments[0].nonce, [p.as_condition_args() for p in payments]))
        asset_pmts.append([asset_id, pmt])
    tail_solution: Program = Program.to([asset_pmts])
    return tail_solution


def get_gopher_puzzle_info() -> PuzzleInfo:
    asset_id = GOPHER_TAIL_PUZZLE_HASH
    driver_item: Dict[str, Any] = {
        "type": AssetType.CAT.value,
        "tail": "0x" + asset_id.hex(),
    }
    return PuzzleInfo(driver_item)  # driver_dict


def get_gopher_mint_amount(xch_amount: uint64) -> uint64:
    return uint64(xch_amount // 10)
