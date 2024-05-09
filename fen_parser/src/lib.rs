use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

const PAWN: u8 = 0;
const KNIGHT: u8 = 1;
const BISHOP: u8 = 2;
const ROOK: u8 = 3;
const QUEEN: u8 = 4;
const KING: u8 = 5;

const WHITE: u8 = 0;
const BLACK: u8 = 6;

fn index(pc_char: u8, rank: u8, file: u8) -> usize {
    let (pt, color) = match pc_char {
        b'K' => (KING, WHITE),
        b'k' => (KING, BLACK),

        b'Q' => (QUEEN, WHITE),
        b'q' => (QUEEN, BLACK),

        b'R' => (ROOK, WHITE),
        b'r' => (ROOK, BLACK),

        b'B' => (BISHOP, WHITE),
        b'b' => (BISHOP, BLACK),

        b'N' => (KNIGHT, WHITE),
        b'n' => (KNIGHT, BLACK),

        b'P' => (PAWN, WHITE),
        b'p' => (PAWN, BLACK),
        _ => {
            panic!("Unrecognized piece type: {}.", pc_char);
        }
    };
    let pc = (color + pt) as usize;
    let sq = ((rank << 3) ^ file) as usize;
    64 * pc + sq
}

#[pyfunction]
fn fen_to_vec(fen: &str) -> PyResult<Vec<u8>> {
    let mut inputs = vec![0; 768];
    let mut rank = 7;
    let mut file = 0;

    let position = match fen.split(' ').next() {
        Some(s) => String::from(s),
        None => panic!("Malformed FEN."),
    };
    for &char in position.as_bytes() {
        match char {
            b'A'..=b'z' => {
                inputs[index(char, rank, file)] = 1;
                file += 1;
            }
            b'/' => {
                rank -= 1;
                file = 0;
            }
            b'1'..=b'8' => {
                file += char - b'0';
            }
            _ => {
                panic!("Unrecognized character.");
            }
        }
    }
    Ok(inputs)
}

#[pymodule]
fn fen_parser(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fen_to_vec, m)?)?;
    Ok(())
}
