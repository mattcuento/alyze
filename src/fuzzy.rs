//! Fuzzy matching with UTF-8 byte ranges and pattern-sized DP scratch space.

use std::ops::Range;

/// Restricts matches to the beginning or end of the haystack.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Anchor {
    #[default]
    None,
    Start,
    End,
}

/// Borrows a prepared pattern so it can be reused across haystacks.
#[derive(Debug, Clone, Copy)]
pub struct Matcher<'a> {
    pattern: &'a [char],
    anchor: Anchor,
    edit_distance: usize,
    case_sensitive: bool,
}

impl<'a> Matcher<'a> {
    pub fn new(
        pattern: &'a [char],
        anchor: Anchor,
        edit_distance: usize,
        case_sensitive: bool,
    ) -> Self {
        Self {
            pattern,
            anchor,
            edit_distance,
            case_sensitive,
        }
    }

    pub fn is_match(self, haystack: &str) -> bool {
        self.candidates(haystack).next().is_some()
    }

    /// Byte ranges of non-overlapping matches in `haystack`, in order, including zero-width matches.
    /// CS101 implementation of fuzzy string matching, using a simple DP approach.
    ///
    /// Order of precedence since returning ranges opens up decisions for match range behavior in overlaps:
    /// 1. At every endpoint, prefer smaller edit distance
    /// 2. At the same endpoint and distance, prefer the longer range (earlier start)
    /// 3. If a candidate match overlaps `pending` or shares its start:
    ///     1. prefer smaller edit distance
    ///     2. prefer the earlier starting match
    ///     3. prefer extending the match if starts are the same
    ///
    /// Keep non-overlapping and touching ranges with different starts separate.
    ///
    /// Complex examples: pattern 'puffin' against 'puffinuffin' with distance 1 returns: [puffin]uffin
    /// Complex examples: pattern 'puffin' against 'uffinuffinuffin' with distance 1 returns: [uffin]uffi[nuffin]
    pub fn find_match_ranges(self, haystack: &str) -> Vec<Range<usize>> {
        let mut matches: Vec<Range<usize>> = Vec::new();
        let mut pending: Option<(Range<usize>, usize)> = None;
        for (range, distance) in self.candidates(haystack) {
            let start = range.start;
            match &pending {
                // replace pending match with fewer edits, or extend from the same start at equal distance
                Some((prev, prev_distance)) if start < prev.end || start == prev.start => {
                    if distance < *prev_distance
                        || (distance == *prev_distance && start == prev.start)
                    {
                        pending = Some((range, distance));
                    }
                }
                // finalize pending match
                _ => {
                    matches.extend(pending.take().map(|(range, _)| range));
                    pending = Some((range, distance));
                }
            }
        }
        matches.extend(pending.map(|(range, _)| range));
        matches
    }

    /// Best in-budget alignment at each matching endpoint, before overlap selection.
    pub fn candidates(self, haystack: &str) -> impl Iterator<Item = (Range<usize>, usize)> {
        // TODO Myers bit-parallel algorithm would be much faster for patterns up to 64 chars,
        // but not worth starting with this until this starts showing up as a bottleneck.
        let m = self.pattern.len();
        let k = self.edit_distance;
        // Fundamentally, DP table where [i][j] = edits to match pattern[0..i] with haystack[0..j]
        // Each cell stores (edit distance, start byte), earlier starts win on equal distance
        let mut dp: Vec<(usize, usize)> = (0..=m).map(|i| (i, 0)).collect();
        let mut matches_empty = match self.anchor {
            Anchor::None | Anchor::Start => m <= k,
            Anchor::End => m <= k && haystack.is_empty(),
        };
        let mut chars = haystack.char_indices().enumerate();
        std::iter::from_fn(move || {
            if std::mem::take(&mut matches_empty) {
                return Some((0..0, m));
            }
            for (j, (byte_idx, haystack_char)) in chars.by_ref() {
                let end = byte_idx + haystack_char.len_utf8();
                let mut prev_diag = dp[0];
                // If we anchor to the start, we _do_ want to penalize for starting later in the haystack.
                match self.anchor {
                    Anchor::Start => dp[0] = (j + 1, 0),
                    Anchor::None | Anchor::End => dp[0] = (0, end),
                }
                for i in 1..=m {
                    let old = dp[i];
                    // Zero cost if chars match, otherwise 1 for a substitution.
                    // old + 1 inserts a haystack char, dp[i-1] + 1 deletes a pattern char.
                    let cost = if self.chars_equivalent(self.pattern[i - 1], haystack_char) {
                        0
                    } else {
                        1
                    };
                    dp[i] = (prev_diag.0 + cost, prev_diag.1)
                        .min((old.0 + 1, old.1))
                        .min((dp[i - 1].0 + 1, dp[i - 1].1));
                    prev_diag = old;
                }
                let (distance, start) = dp[m];
                // Without an anchor, a match can end anywhere in the haystack.
                // With an end anchor, only the final haystack position matters.
                let matched = match self.anchor {
                    Anchor::End => end == haystack.len() && distance <= k,
                    Anchor::None | Anchor::Start => distance <= k,
                };
                if matched {
                    return Some((start..end, distance));
                }
            }
            None
        })
    }

    /// Handles `char` comparison between `l` and `r` for edit distance calculation, handling case
    /// sensitivity via `to_lowercase()` if applicable.
    ///
    /// In case-insensitive situations, unicode characters are NOT case-folded in this
    /// implementation. Multi-scalar expansion is handled in identity cases only via use of `eq`
    /// with `ToLowercase` iterator.
    fn chars_equivalent(&self, l: char, r: char) -> bool {
        // todo (cuento): eventually would be nice to use ASCII case-folding here like
        //      in our analyzer pipeline for equivalent prefiltering/postfiltering behavior
        if self.case_sensitive {
            l == r
        } else {
            l.to_lowercase().eq(r.to_lowercase())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn match_ranges() {
        let cases = [
            (
                "puffin",
                "puffinpuffin",
                Anchor::None,
                0,
                true,
                vec![0..6, 6..12],
            ),
            ("puffin", "puffinuffin", Anchor::None, 1, true, vec![0..6]),
            (
                "puffin",
                "uffinuffinuffin",
                Anchor::None,
                1,
                true,
                vec![0..5, 9..15],
            ),
            ("a", "a", Anchor::None, 1, true, vec![0..1]),
            ("a", "a", Anchor::Start, 1, true, vec![0..1]),
            ("a", "", Anchor::End, 1, true, vec![0..0]),
            ("", "aé", Anchor::None, 0, true, vec![0..0, 1..1, 3..3]),
            ("", "aé", Anchor::Start, 1, true, vec![0..0]),
            ("é", "☕ É!", Anchor::None, 0, false, vec![4..6]),
            ("bird", "bird bird", Anchor::Start, 0, true, vec![0..4]),
            ("bird", "bird bird", Anchor::End, 0, true, vec![5..9]),
        ];
        for (pattern, text, anchor, distance, case_sensitive, expected) in cases {
            let chars: Vec<char> = pattern.chars().collect();
            let matcher = Matcher::new(&chars, anchor, distance, case_sensitive);
            assert_eq!(
                matcher.find_match_ranges(text),
                expected,
                "pattern={pattern:?}, text={text:?}"
            );
            assert_eq!(matcher.is_match(text), !expected.is_empty());
        }
    }

    #[test]
    fn boolean_matching_stops_at_first_candidate() {
        let pattern: Vec<char> = "puffin".chars().collect();
        let matcher = Matcher::new(&pattern, Anchor::None, 1, true);
        assert_eq!(matcher.candidates("puffin").next(), Some((0..5, 1)));
        assert!(matcher.is_match("puffin"));
    }
}
