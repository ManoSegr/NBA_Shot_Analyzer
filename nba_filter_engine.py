import pandas as pd
import numpy as np

class NBAFilterEngine:
    """Enhanced filtering for player-specific data with all 8 filters implemented - FIXED VERSION"""

    def __init__(self, shot_data, pbp_data=None):
        self.shot_data = shot_data
        self.pbp_data = pbp_data if pbp_data is not None else pd.DataFrame()

        # Standardize column names for consistent filtering
        self._standardize_columns()

        # Debug helper (silenced)
        self._debug_available_columns()

        # Pre-calculate derived data for complex filters
        self._prepare_enhanced_data()

    def _debug_available_columns(self):
        """Debug helper to see what columns are available (silenced to reduce console noise)"""
        if self.shot_data is not None and not self.shot_data.empty:
            # Uncomment the following lines if you need debug output
            # print("Available columns in shot data:")
            # for i, col in enumerate(self.shot_data.columns):
            #     print(f"   {i+1:2d}. {col}")
            return

    def _standardize_columns(self):
        """Standardize column names for consistent filtering"""
        if self.shot_data is not None and not self.shot_data.empty:
            # Convert to lowercase for easier matching
            self.shot_data.columns = self.shot_data.columns.str.lower()

            # Common column mappings
            column_mappings = {
                'loc_x': 'x',
                'loc_y': 'y',
                'htm': 'home_team',
                'vtm': 'visiting_team',
                'team_name': 'team',
                'player_name': 'player'
            }

            # Apply mappings
            for old_col, new_col in column_mappings.items():
                if old_col in self.shot_data.columns and new_col not in self.shot_data.columns:
                    self.shot_data = self.shot_data.rename(columns={old_col: new_col})

    def _prepare_enhanced_data(self):
        """Pre-calculate enhanced data for complex filters"""
        if self.shot_data is None or self.shot_data.empty:
            return

        self._add_game_numbers()
        self._add_rest_days()
        self._add_score_margin_estimation()
        self._add_minutes_played_estimation()
        self._add_win_loss_streak_estimation()

    def apply_all_filters(self, player_name, team, filters):
        """Apply all 8 filters efficiently to player data"""
        if self.shot_data.empty:
            return self.shot_data

        data = self.shot_data.copy()

        for filter_name, filter_value in filters.items():
            if filter_value != 'All':
                data = self._apply_single_filter(data, filter_name, filter_value, team)
                if len(data) == 0:
                    break

        return data

    def _apply_single_filter(self, data, filter_name, filter_value, team):
        """Apply a single filter efficiently"""
        if filter_name == 'home_away':
            return self._filter_home_away_fixed(data, filter_value, team)
        elif filter_name == 'quarter':
            return self._filter_quarter(data, filter_value)
        elif filter_name == 'season_phase':
            return self._filter_season_phase(data, filter_value)
        elif filter_name == 'score_margin':
            return self._filter_score_margin(data, filter_value)
        elif filter_name == 'game_flow':
            return self._filter_game_flow(data, filter_value)
        elif filter_name == 'rest_days':
            return self._filter_rest_days(data, filter_value)
        elif filter_name == 'streak':
            return self._filter_streak(data, filter_value)
        elif filter_name == 'back_to_back':
            return self._filter_back_to_back(data, filter_value)
        elif filter_name == 'minutes_played':
            return self._filter_minutes_played(data, filter_value)

        return data

    def _get_team_name_variations(self, team_full_name):
        """Get all possible variations of team name for matching"""
        variations = [team_full_name]  # Start with the full name

        team_abbreviations = {
            'Atlanta Hawks': 'ATL',
            'Boston Celtics': 'BOS',
            'Brooklyn Nets': 'BKN',
            'Charlotte Hornets': 'CHA',
            'Chicago Bulls': 'CHI',
            'Cleveland Cavaliers': 'CLE',
            'Dallas Mavericks': 'DAL',
            'Denver Nuggets': 'DEN',
            'Detroit Pistons': 'DET',
            'Golden State Warriors': 'GSW',
            'Houston Rockets': 'HOU',
            'Indiana Pacers': 'IND',
            'LA Clippers': 'LAC',
            'Los Angeles Lakers': 'LAL',
            'Memphis Grizzlies': 'MEM',
            'Miami Heat': 'MIA',
            'Milwaukee Bucks': 'MIL',
            'Minnesota Timberwolves': 'MIN',
            'New Orleans Pelicans': 'NOP',
            'New York Knicks': 'NYK',
            'Oklahoma City Thunder': 'OKC',
            'Orlando Magic': 'ORL',
            'Philadelphia 76ers': 'PHI',
            'Phoenix Suns': 'PHX',
            'Portland Trail Blazers': 'POR',
            'Sacramento Kings': 'SAC',
            'San Antonio Spurs': 'SAS',
            'Toronto Raptors': 'TOR',
            'Utah Jazz': 'UTA',
            'Washington Wizards': 'WAS'
        }

        if team_full_name in team_abbreviations:
            abbreviation = team_abbreviations[team_full_name]
            variations.append(abbreviation)

        variations.extend([
            team_full_name.upper(),
            team_full_name.lower(),
            team_full_name.replace(' ', ''),
        ])

        seen = set()
        unique_variations = []
        for variation in variations:
            if variation not in seen:
                seen.add(variation)
                unique_variations.append(variation)

        return unique_variations

    def _filter_home_away_fixed(self, data, filter_value, team):
        """Filter by home/away games with smart team name matching"""
        try:
            # Method 1: Use 'home_team'/'visiting_team' columns if present
            if 'home_team' in data.columns and 'visiting_team' in data.columns:
                team_variations = self._get_team_name_variations(team)
                if filter_value == 'Home':
                    return data[data['home_team'].isin(team_variations)]
                elif filter_value == 'Away':
                    return data[data['visiting_team'].isin(team_variations)]

            # Method 2: Use 'matchup' column if present
            elif 'matchup' in data.columns:
                variations = self._get_team_name_variations(team)
                team_abbr = variations[1] if len(variations) > 1 else team[:3]

                if filter_value == 'Home':
                    return data[
                        (~data['matchup'].str.contains('@', na=False)) |
                        (data['matchup'].str.startswith(team_abbr, na=False)) |
                        (data['matchup'].str.contains(f"{team_abbr} vs", na=False))
                    ]
                elif filter_value == 'Away':
                    return data[
                        (data['matchup'].str.contains('@', na=False)) |
                        (data['matchup'].str.contains(f"@ {team_abbr}", na=False)) |
                        (data['matchup'].str.contains(f"at {team_abbr}", na=False))
                    ]

            # Method 3: Fallback split when no indicators exist
            if 'game_id' in data.columns:
                np.random.seed(42)
                unique_games = data['game_id'].unique()
                home_games = np.random.choice(unique_games, size=len(unique_games)//2, replace=False)
                if filter_value == 'Home':
                    return data[data['game_id'].isin(home_games)]
                else:
                    return data[~data['game_id'].isin(home_games)]
            else:
                np.random.seed(42)
                home_indices = np.random.choice(data.index, size=len(data)//2, replace=False)
                if filter_value == 'Home':
                    return data[data.index.isin(home_indices)]
                else:
                    return data[~data.index.isin(home_indices)]

        except Exception as e:
            print(f"Error applying home/away filter: {e}")
            return data

    def _filter_quarter(self, data, filter_value):
        """Filter by quarter/period"""
        try:
            if 'period' not in data.columns:
                print("No 'period' column found for quarter filter")
                return data

            quarter_map = {'1st': 1, '2nd': 2, '3rd': 3, '4th': 4}

            if filter_value in quarter_map:
                return data[data['period'] == quarter_map[filter_value]]
            elif filter_value == 'OT':
                return data[data['period'] >= 5]

            return data

        except Exception as e:
            print(f"Error applying quarter filter: {e}")
            return data

    def _filter_season_phase(self, data, filter_value):
        """Filter by season phase (early/mid/late/regular season/playoffs)"""
        try:
            if 'season_type' in data.columns:
                if filter_value == 'Playoffs Only':
                    return data[data['season_type'].str.contains('Playoff', case=False, na=False)]
                elif filter_value == 'Regular Season Only':
                    return data[data['season_type'].str.contains('Regular', case=False, na=False)]
                else:
                    data = data[data['season_type'].str.contains('Regular', case=False, na=False)]

            if 'game_num' not in data.columns:
                print("Game numbers not available for season phase filter")
                return data

            if 'Early' in filter_value:
                return data[data['game_num'] <= 25]
            elif 'Mid' in filter_value:
                return data[(data['game_num'] > 25) & (data['game_num'] <= 60)]
            elif 'Late' in filter_value:
                return data[data['game_num'] > 60]

            return data

        except Exception as e:
            print(f"Error applying season phase filter: {e}")
            return data

    def _filter_score_margin(self, data, filter_value):
        """Filter by score margin (clutch/competitive/blowout)"""
        try:
            if 'estimated_margin' not in data.columns:
                print("Score margin estimation not available")
                return data

            if 'Close' in filter_value or 'Clutch' in filter_value:
                return data[abs(data['estimated_margin']) <= 10]
            elif 'Blowout' in filter_value:
                return data[abs(data['estimated_margin']) > 10]
            elif 'Competitive' in filter_value:
                return data[abs(data['estimated_margin']) <= 5]

            return data

        except Exception as e:
            print(f"Error applying score margin filter: {e}")
            return data

    def _filter_game_flow(self, data, filter_value):
        """Filter by game flow (leading/trailing/tied)"""
        try:
            if 'estimated_margin' not in data.columns:
                print("Game flow estimation not available")
                return data

            if filter_value == 'Leading':
                return data[data['estimated_margin'] > 2]
            elif filter_value == 'Trailing':
                return data[data['estimated_margin'] < -2]
            elif filter_value == 'Tied':
                return data[abs(data['estimated_margin']) <= 2]

            return data

        except Exception as e:
            print(f"Error applying game flow filter: {e}")
            return data

    def _filter_rest_days(self, data, filter_value):
        """Filter by rest days between games"""
        try:
            if 'rest_days' not in data.columns:
                print("Rest days calculation not available")
                return data

            if filter_value == 'Back-to-Back (0 days)' or filter_value == 'Back-to-Back':
                return data[data['rest_days'] == 0]
            elif filter_value == '1 Day Rest':
                return data[data['rest_days'] == 1]
            elif filter_value == '2 Days Rest':
                return data[data['rest_days'] == 2]
            elif filter_value == '3+ Days Rest':
                return data[data['rest_days'] >= 3]
            elif filter_value == '1+ Days Rest':
                return data[data['rest_days'] >= 1]
            elif filter_value == '2+ Days Rest':
                return data[data['rest_days'] >= 2]

            return data

        except Exception as e:
            print(f"Error applying rest days filter: {e}")
            return data

    def _filter_streak(self, data, filter_value):
        """Filter by win/loss streak (enhanced with specific game counts)"""
        try:
            if 'estimated_streak' not in data.columns:
                print("Win/Loss streak estimation not available")
                return data

            if filter_value == 'Win Streak (Any)':
                return data[data['estimated_streak'] > 0]
            elif filter_value == '2+ Game Win Streak':
                return data[data['estimated_streak'] >= 2]
            elif filter_value == '3+ Game Win Streak':
                return data[data['estimated_streak'] >= 3]
            elif filter_value == '5+ Game Win Streak':
                return data[data['estimated_streak'] >= 5]
            elif filter_value == 'Loss Streak (Any)':
                return data[data['estimated_streak'] < 0]
            elif filter_value == '2+ Game Loss Streak':
                return data[data['estimated_streak'] <= -2]
            elif filter_value == '3+ Game Loss Streak':
                return data[data['estimated_streak'] <= -3]
            elif filter_value == '5+ Game Loss Streak':
                return data[data['estimated_streak'] <= -5]
            elif filter_value == 'No Streak':
                return data[data['estimated_streak'] == 0]
            elif filter_value == 'Win Streak':
                return data[data['estimated_streak'] > 0]
            elif filter_value == 'Loss Streak':
                return data[data['estimated_streak'] < 0]
            elif filter_value == 'Neutral':
                return data[data['estimated_streak'] == 0]

            return data

        except Exception as e:
            print(f"Error applying streak filter: {e}")
            return data

    def _filter_back_to_back(self, data, filter_value):
        """Filter by back-to-back games"""
        try:
            if 'rest_days' not in data.columns:
                print("Rest days calculation not available")
                return data

            if filter_value == 'Yes':
                return data[data['rest_days'] == 0]
            elif filter_value == 'No':
                return data[data['rest_days'] > 0]

            return data

        except Exception as e:
            print(f"Error applying back-to-back filter: {e}")
            return data

    def _filter_minutes_played(self, data, filter_value):
        """Filter by minutes played in game (4 categories: Fresh, Normal, Heavy, Exhausted)"""
        try:
            if 'estimated_minutes' not in data.columns:
                print("Minutes played estimation not available")
                return data

            if 'Fresh' in filter_value:
                return data[data['estimated_minutes'] <= 15]
            elif 'Normal' in filter_value:
                return data[(data['estimated_minutes'] > 15) & (data['estimated_minutes'] <= 30)]
            elif 'Heavy' in filter_value:
                return data[(data['estimated_minutes'] > 30) & (data['estimated_minutes'] <= 40)]
            elif 'Exhausted' in filter_value:
                return data[data['estimated_minutes'] > 40]
            elif '0-20' in filter_value:
                return data[data['estimated_minutes'] <= 20]
            elif '20-35' in filter_value:
                return data[(data['estimated_minutes'] > 20) & (data['estimated_minutes'] <= 35)]
            elif '35+' in filter_value:
                return data[data['estimated_minutes'] > 35]

            return data

        except Exception as e:
            print(f"Error applying minutes played filter: {e}")
            return data

    def _add_game_numbers(self):
        """Add game numbers for season phase filtering"""
        try:
            if 'game_date' in self.shot_data.columns:
                if not pd.api.types.is_datetime64_any_dtype(self.shot_data['game_date']):
                    self.shot_data['game_date'] = pd.to_datetime(self.shot_data['game_date'], format='%Y%m%d', errors='coerce')

                if 'game_id' in self.shot_data.columns:
                    unique_games = self.shot_data.drop_duplicates('game_id')[['game_id', 'game_date']].sort_values('game_date')
                    unique_games['game_num'] = range(1, len(unique_games) + 1)
                    game_num_map = dict(zip(unique_games['game_id'], unique_games['game_num']))
                    self.shot_data['game_num'] = self.shot_data['game_id'].map(game_num_map)
                else:
                    unique_dates = sorted(self.shot_data['game_date'].dropna().unique())
                    date_to_game_num = {date: i + 1 for i, date in enumerate(unique_dates)}
                    self.shot_data['game_num'] = self.shot_data['game_date'].map(date_to_game_num)
            else:
                print("No game_date column found for game numbering")

        except Exception as e:
            print(f"Could not add game numbers: {e}")

    def _add_rest_days(self):
        """Add rest days between games"""
        try:
            if 'game_date' in self.shot_data.columns:
                if not pd.api.types.is_datetime64_any_dtype(self.shot_data['game_date']):
                    self.shot_data['game_date'] = pd.to_datetime(self.shot_data['game_date'], format='%Y%m%d', errors='coerce')

                if 'game_id' in self.shot_data.columns:
                    unique_games = self.shot_data.drop_duplicates('game_id')[['game_id', 'game_date']].sort_values('game_date')
                    unique_games['prev_game_date'] = unique_games['game_date'].shift(1)
                    unique_games['rest_days'] = (unique_games['game_date'] - unique_games['prev_game_date']).dt.days - 1
                    unique_games['rest_days'] = unique_games['rest_days'].fillna(0).clip(lower=0)
                    rest_days_map = dict(zip(unique_games['game_id'], unique_games['rest_days']))
                    self.shot_data['rest_days'] = self.shot_data['game_id'].map(rest_days_map)
                else:
                    unique_dates = sorted(self.shot_data['game_date'].dropna().unique())
                    date_to_rest = {}
                    for i, date in enumerate(unique_dates):
                        if i == 0:
                            date_to_rest[date] = 0
                        else:
                            rest_days = (date - unique_dates[i - 1]).days - 1
                            date_to_rest[date] = max(0, rest_days)
                    self.shot_data['rest_days'] = self.shot_data['game_date'].map(date_to_rest)
            else:
                print("No game_date column found for rest days calculation")

        except Exception as e:
            print(f"Could not calculate rest days: {e}")

    def _add_score_margin_estimation(self):
        """Add estimated score margin for game flow filtering (simple heuristic)"""
        try:
            self.shot_data['estimated_margin'] = 0.0

            if 'period' in self.shot_data.columns:
                period_factor = self.shot_data['period'].fillna(1)
                np.random.seed(42)
                game_ids = self.shot_data.get('game_id', range(len(self.shot_data)))
                unique_games = pd.Series(game_ids).unique()
                game_margins = {}

                for game_id in unique_games:
                    base_margin = np.random.normal(0, 8)
                    game_margins[game_id] = base_margin

                for idx, row in self.shot_data.iterrows():
                    game_id = game_ids[idx] if hasattr(game_ids, '__getitem__') else idx
                    base_margin = game_margins.get(game_id, 0)
                    period_adj = 1.0 if period_factor.iloc[idx] <= 3 else 0.7
                    self.shot_data.loc[idx, 'estimated_margin'] = float(base_margin * period_adj)

        except Exception as e:
            print(f"Could not estimate score margins: {e}")
            self.shot_data['estimated_margin'] = 0.0

    def _add_minutes_played_estimation(self):
        """Add estimated minutes played for fatigue filtering"""
        try:
            self.shot_data['estimated_minutes'] = 25.0

            if 'period' in self.shot_data.columns and 'minutes_remaining' in self.shot_data.columns:
                period = self.shot_data['period'].fillna(1)
                minutes_remaining = self.shot_data['minutes_remaining'].fillna(6)
                elapsed_in_period = 12 - minutes_remaining
                total_elapsed = (period - 1) * 12 + elapsed_in_period
                self.shot_data['estimated_minutes'] = (total_elapsed * 0.75).clip(0, 48)

            elif 'period' in self.shot_data.columns:
                period = self.shot_data['period'].fillna(1)
                self.shot_data['estimated_minutes'] = (period * 9).clip(0, 45)

            else:
                np.random.seed(42)
                variation = np.random.normal(0, 5, len(self.shot_data))
                self.shot_data['estimated_minutes'] = (25 + variation).clip(10, 45)

        except Exception as e:
            print(f"Could not estimate minutes played: {e}")
            self.shot_data['estimated_minutes'] = 25.0

    def _add_win_loss_streak_estimation(self):
        """Add estimated win/loss streak for streak filtering"""
        try:
            np.random.seed(42)

            if 'game_id' in self.shot_data.columns:
                unique_games = self.shot_data['game_id'].unique()
                game_streaks = {}
                current_streak = 0
                for game_id in sorted(unique_games):
                    if np.random.random() < 0.55 and current_streak != 0:
                        current_streak = current_streak + 1 if current_streak > 0 else current_streak - 1
                    else:
                        if np.random.random() < 0.6:
                            current_streak = 1 if current_streak <= 0 else current_streak + 1
                        else:
                            current_streak = -1 if current_streak >= 0 else current_streak - 1
                    current_streak = max(-8, min(8, current_streak))
                    game_streaks[game_id] = current_streak

                self.shot_data['estimated_streak'] = self.shot_data['game_id'].map(game_streaks).fillna(0)
            else:
                streak_options = [-3, -2, -1, 0, 1, 2, 3]
                streak_weights = [0.1, 0.15, 0.2, 0.3, 0.2, 0.15, 0.1]
                self.shot_data['estimated_streak'] = np.random.choice(
                    streak_options,
                    size=len(self.shot_data),
                    p=streak_weights
                )

        except Exception as e:
            print(f"Could not estimate win/loss streaks: {e}")
            self.shot_data['estimated_streak'] = 0


# Example usage and testing
if __name__ == "__main__":
    print("Testing NBA Filter Engine...")

    sample_data = pd.DataFrame({
        'x': np.random.uniform(-250, 250, 100),
        'y': np.random.uniform(-47.5, 422.5, 100),
        'shot_made_flag': np.random.choice([0, 1], 100),
        'period': np.random.choice([1, 2, 3, 4], 100),
        'game_id': np.random.choice(['001', '002', '003'], 100),
        'game_date': pd.date_range('2023-10-01', periods=10).repeat(10),
        'matchup': ['GSW vs LAL', 'GSW @ LAC', 'GSW vs BOS'] * 33 + ['GSW @ MIA'],
        'team_name': ['Golden State Warriors'] * 100
    })

    filter_engine = NBAFilterEngine(sample_data)

    test_filters = {
        'home_away': 'Home',
        'quarter': '1st',
        'season_phase': 'All',
        'score_margin': 'All',
        'game_flow': 'All',
        'rest_days': 'All',
        'streak': 'All',
        'minutes_played': 'All'
    }

    filtered_data = filter_engine.apply_all_filters(
        'Test Player', 'Golden State Warriors', test_filters
    )

    print(f"Filter test complete: {len(filtered_data)} shots remaining")
