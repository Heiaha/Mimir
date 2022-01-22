use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use chess::{Board, Piece, Color};
use std::str::FromStr;

#[pyfunction]
fn fen_to_vec(fen: &str) -> PyResult<Vec<u8>> {
    let mut vector = vec![0u8; 768];
    let board = Board::from_str(fen).expect("Invalid fen.");
    let white = board.color_combined(Color::White);
    let black = board.color_combined(Color::Black);
    let bitboards = [
        board.pieces(Piece::Pawn) & white,
        board.pieces(Piece::Knight) & white,
        board.pieces(Piece::Bishop) & white,
        board.pieces(Piece::Rook) & white,
        board.pieces(Piece::Queen) & white,
        board.pieces(Piece::King) & white,
        board.pieces(Piece::Pawn) & black,
        board.pieces(Piece::Knight) & black,
        board.pieces(Piece::Bishop) & black,
        board.pieces(Piece::Rook) & black,
        board.pieces(Piece::Queen) & black,
        board.pieces(Piece::King) & black,
    ];

    for (index, piece_bb) in bitboards.iter().enumerate() {
        for sq in *piece_bb {
            vector[index*64 + sq.to_int() as usize] = 1;
        }
    }
    Ok(vector)
}

#[pymodule]
fn fen_parser(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fen_to_vec, m)?)?;
    Ok(())
}