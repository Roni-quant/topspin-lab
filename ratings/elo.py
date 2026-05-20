"""Elo rating engine for table tennis match outcomes.

Implements the standard Elo rating formula with configurable K-factor.
All ratings are updated after each match in strict chronological order.
"""

from dataclasses import dataclass


@dataclass
class EloConfig:
    """Configuration for Elo rating calculations."""
    base_rating: float = 1500.0
    k_factor: float = 32.0  # Standard K-factor for individual players


def expected_score(elo_a: float, elo_b: float) -> float:
    """Calculate expected winning probability for player A.

    Args:
        elo_a: Elo rating of player A
        elo_b: Elo rating of player B

    Returns:
        Probability (0-1) that player A wins
    """
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def update_rating(
    elo: float,
    expected: float,
    actual: float,
    k_factor: float = 32.0,
) -> float:
    """Update Elo rating after a match.

    Args:
        elo: Current Elo rating
        expected: Expected score (0-1) from expected_score()
        actual: Actual score (1 = win, 0 = loss)
        k_factor: K-factor determining rating volatility

    Returns:
        New Elo rating
    """
    return elo + k_factor * (actual - expected)


class EloRatingEngine:
    """Maintains player ratings and updates them match-by-match."""

    def __init__(self, config: EloConfig | None = None):
        """Initialize the Elo engine.

        Args:
            config: EloConfig instance (uses defaults if None)
        """
        self.config = config or EloConfig()
        self.ratings: dict[int, float] = {}

    def get_rating(self, player_id: int) -> float:
        """Get current rating for a player, initializing at base if needed."""
        if player_id not in self.ratings:
            self.ratings[player_id] = self.config.base_rating
        return self.ratings[player_id]

    def process_match(
        self,
        player_a_id: int,
        player_b_id: int,
        winner_id: int,
    ) -> tuple[float, float]:
        """Process a single match and update ratings.

        Args:
            player_a_id: Player A's ID
            player_b_id: Player B's ID
            winner_id: ID of the winning player

        Returns:
            Tuple of (elo_a_before, elo_b_before)
        """
        # Get ratings before the match
        elo_a = self.get_rating(player_a_id)
        elo_b = self.get_rating(player_b_id)

        # Calculate expected scores
        exp_a = expected_score(elo_a, elo_b)
        exp_b = 1.0 - exp_a

        # Determine actual scores
        actual_a = 1.0 if winner_id == player_a_id else 0.0
        actual_b = 1.0 if winner_id == player_b_id else 0.0

        # Update ratings
        new_elo_a = update_rating(elo_a, exp_a, actual_a, self.config.k_factor)
        new_elo_b = update_rating(elo_b, exp_b, actual_b, self.config.k_factor)

        self.ratings[player_a_id] = new_elo_a
        self.ratings[player_b_id] = new_elo_b

        return elo_a, elo_b

    def get_all_ratings(self) -> dict[int, float]:
        """Return a copy of all current ratings."""
        return self.ratings.copy()
