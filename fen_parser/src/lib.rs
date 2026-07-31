use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

const PIECES: &[u8] = b"PNBRQKpnbrqk";
const WHITE_KING: usize = 5;
const BLACK_KING: usize = 11;

const N_FILES: usize = 8;
const N_RANKS: usize = 8;
const N_SQUARES: usize = N_FILES * N_RANKS;
const N_PIECE_CODES: usize = PIECES.len();
const COLOR_FLIP: usize = N_PIECE_CODES / 2;
const FILE_MIRROR: usize = N_FILES - 1;
const RANK_FLIP: usize = 56;
const MAX_PIECES: usize = 32;

const N_BASE_FEATURES: usize = N_PIECE_CODES * N_SQUARES;
const N_KING_BUCKETS: usize = 4;
const N_FEATURES: usize = N_KING_BUCKETS * N_BASE_FEATURES;

fn king_bucket(ksq_norm: usize) -> usize {
    match ksq_norm / N_FILES {
        0 => 0,
        1 => 1,
        2 | 3 => 2,
        _ => 3,
    }
}

enum Color {
    White,
    Black,
}

struct Perspective {
    sq_xor: usize,
    pc_flip: usize,
    offset: usize,
}

impl Perspective {
    fn new(color: Color, own_king_sq: usize) -> Self {
        let (rel_xor, pc_flip) = match color {
            Color::White => (0, 0),
            Color::Black => (RANK_FLIP, COLOR_FLIP),
        };
        let ksq_rel = own_king_sq ^ rel_xor;
        let mirror_xor = if ksq_rel % N_FILES >= N_FILES / 2 {
            FILE_MIRROR
        } else {
            0
        };

        Self {
            sq_xor: rel_xor ^ mirror_xor,
            pc_flip,
            offset: N_BASE_FEATURES * king_bucket(ksq_rel ^ mirror_xor),
        }
    }

    fn feature_index(&self, pc: usize, sq: usize) -> usize {
        let index = self.offset
            + N_SQUARES * ((pc + self.pc_flip) % N_PIECE_CODES)
            + (sq ^ self.sq_xor);
        assert!(index < N_FEATURES);
        index
    }
}

// Side-to-move perspective only; the trainer derives nstm from these.
fn parse_indices(fen: &str) -> Vec<usize> {
    let mut pieces: Vec<(usize, usize)> = Vec::with_capacity(MAX_PIECES);
    let mut rank: usize = N_RANKS - 1;
    let mut file: usize = 0;

    let mut parts = fen.split_whitespace();

    let position = parts.next().expect("Malformed FEN.");

    let stm = match parts.next() {
        Some("w") => Color::White,
        Some("b") => Color::Black,
        _ => panic!("Unrecognized color."),
    };

    for &char in position.as_bytes() {
        match char {
            b'A'..=b'z' => {
                let pc = PIECES
                    .iter()
                    .position(|&pc_char| pc_char == char)
                    .unwrap_or_else(|| panic!("Unrecognized piece"));
                let sq = rank * N_FILES + file;
                pieces.push((pc, sq));
                file += 1;
            }
            b'/' => {
                rank -= 1;
                file = 0;
            }
            b'1'..=b'8' => {
                file += (char - b'0') as usize;
            }
            _ => {
                panic!("Unrecognized character.");
            }
        }
    }

    let own_king = match stm {
        Color::White => WHITE_KING,
        Color::Black => BLACK_KING,
    };
    let king_sq = pieces
        .iter()
        .find_map(|&(pc, sq)| (pc == own_king).then_some(sq))
        .expect("FEN has no king.");

    let own = Perspective::new(stm, king_sq);

    let mut stm_indices: Vec<usize> = pieces
        .iter()
        .map(|&(pc, sq)| own.feature_index(pc, sq))
        .collect();

    stm_indices.resize(MAX_PIECES, N_FEATURES);
    stm_indices
}

#[pyfunction]
fn fen_to_indices(fen: &str) -> PyResult<Vec<usize>> {
    Ok(parse_indices(fen))
}

#[pyfunction]
fn indices_to_vec(indices: Vec<usize>) -> PyResult<[f32; N_FEATURES]> {
    let mut input = [0.0; N_FEATURES];

    for idx in indices {
        input[idx] = 1.0
    }

    Ok(input)
}

#[pymodule]
fn fen_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fen_to_indices, m)?)?;
    m.add_function(wrap_pyfunction!(indices_to_vec, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // These indices are hardcoded in Weiawaga's nnue.rs too, in
    // feature_indices_match_trainer_convention. Both must change together.
    #[test]
    fn feature_indices_match_engine_convention() {
        // White king b1, not mirrored; black king g8, mirrored to b1. Both in
        // bucket 0, and the FEN scan reaches the black king first.
        let white = parse_indices("6k1/8/8/8/8/8/8/1K6 w - - 0 1");
        let black = parse_indices("6k1/8/8/8/8/8/8/1K6 b - - 0 1");

        assert_eq!(&white[..2], &[766, 321]);
        assert_eq!(&black[..2], &[321, 766]);
        assert!(
            white[2..]
                .iter()
                .chain(black[2..].iter())
                .all(|&i| i == N_FEATURES)
        );
    }

    #[test]
    fn bucket_offsets_match_engine_convention() {
        // White king d5 (rank 5, bucket 3), black king g8 (mirrored to b1,
        // bucket 0).
        let white = parse_indices("6k1/8/8/3K4/8/8/8/8 w - - 0 1");
        let black = parse_indices("6k1/8/8/3K4/8/8/8/8 b - - 0 1");

        assert_eq!(&white[..2], &[3070, 2659]);
        assert_eq!(&black[..2], &[321, 732]);
    }

    #[test]
    fn startpos_perspectives_are_symmetric() {
        // Both kings on the e-file, so both perspectives mirror: the two
        // index sets must come out equal.
        let mut white =
            parse_indices("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let mut black =
            parse_indices("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");

        white.sort_unstable();
        black.sort_unstable();
        assert_eq!(white, black);
    }
}
