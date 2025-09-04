import pandas as pd
import numpy as np
import os
import sys
from functools import lru_cache
import time

# NBA API imports
try:
    from nba_api.stats.static import teams, players
    from nba_api.stats.endpoints import commonteamroster
    NBA_API_AVAILABLE = True
    # Silenced success print
except ImportError as e:
    print(f"NBA API not available: {e}")
    print("Install with: pip install nba-api")
    NBA_API_AVAILABLE = False


def get_exe_safe_path(relative_path):
    """Get the correct path whether running as script or EXE"""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS  # Running as EXE - files are in temporary directory
        full_path = os.path.join(base_path, relative_path)
        return full_path
    else:
        # Running as script - files are in current directory
        return relative_path


class EnhancedNBADataManager:
    """Enhanced data manager with full team names and EXE support"""

    def __init__(self, data_dir='nba_data'):
        # Initialize base manager with EXE-safe path
        self.base_manager = NBADataManager(data_dir)

        # Enhanced team mapping with full names
        self.full_team_names = {
            'ATL': 'Atlanta Hawks',
            'BOS': 'Boston Celtics',
            'BKN': 'Brooklyn Nets',
            'CHA': 'Charlotte Hornets',
            'CHI': 'Chicago Bulls',
            'CLE': 'Cleveland Cavaliers',
            'DAL': 'Dallas Mavericks',
            'DEN': 'Denver Nuggets',
            'DET': 'Detroit Pistons',
            'GSW': 'Golden State Warriors',
            'HOU': 'Houston Rockets',
            'IND': 'Indiana Pacers',
            'LAC': 'LA Clippers',
            'LAL': 'Los Angeles Lakers',
            'MEM': 'Memphis Grizzlies',
            'MIA': 'Miami Heat',
            'MIL': 'Milwaukee Bucks',
            'MIN': 'Minnesota Timberwolves',
            'NOP': 'New Orleans Pelicans',
            'NYK': 'New York Knicks',
            'OKC': 'Oklahoma City Thunder',
            'ORL': 'Orlando Magic',
            'PHI': 'Philadelphia 76ers',
            'PHX': 'Phoenix Suns',
            'POR': 'Portland Trail Blazers',
            'SAC': 'Sacramento Kings',
            'SAS': 'San Antonio Spurs',
            'TOR': 'Toronto Raptors',
            'UTA': 'Utah Jazz',
            'WAS': 'Washington Wizards'
        }

        # Reverse mapping for lookups
        self.abbreviation_lookup = {v: k for k, v in self.full_team_names.items()}

    def get_teams_for_season_with_full_names(self, season):
        """Get teams with full names for display"""
        abbreviations = self.base_manager.get_teams_for_season(season)
        full_names = []
        for abbr in abbreviations:
            if abbr in self.full_team_names:
                full_names.append(self.full_team_names[abbr])
            else:
                full_names.append(abbr)  # Fallback to abbreviation
        return sorted(full_names)

    def get_abbreviation_from_full_name(self, full_name):
        """Get abbreviation from full team name"""
        return self.abbreviation_lookup.get(full_name, full_name)

    def get_players_for_team_season(self, season, team_full_name):
        """Get players using full team name"""
        team_abbr = self.get_abbreviation_from_full_name(team_full_name)
        return self.base_manager.get_players_for_team_season(season, team_abbr)

    def load_player_shots(self, season, player_name, include_playoffs=True):
        """Load player shots (pass through to base manager)"""
        return self.base_manager.load_player_shots(season, player_name, include_playoffs)

    # Pass through other methods
    def get_available_seasons(self):
        return self.base_manager.get_available_seasons()

    def validate_data_availability(self, season):
        return self.base_manager.validate_data_availability(season)


class NBADataManager:
    """EXE-Safe NBA Data Manager with NBA API roster integration"""

    def __init__(self, data_dir='nba_data'):
        # Use EXE-safe path
        self.data_dir = get_exe_safe_path(data_dir)

        # Check if data directory exists, try alternatives
        if not os.path.exists(self.data_dir):
            print(f"Data directory not found: {self.data_dir}")

            alternatives = [
                'nba_data',
                './nba_data',
                os.path.join(os.getcwd(), 'nba_data'),
                'datasets',
                './datasets'
            ]

            if hasattr(sys, '_MEIPASS'):
                alternatives.extend([
                    os.path.join(sys._MEIPASS, 'datasets'),
                    os.path.join(sys._MEIPASS, 'nba_data', 'datasets')
                ])

            found = False
            for alt_path in alternatives:
                if os.path.exists(alt_path):
                    print(f"Found data at alternative path: {alt_path}")
                    self.data_dir = alt_path
                    found = True
                    break
            if not found:
                print("No data directory found in any known location.")

        # Initialize caches
        self.metadata_cache = {}
        self.roster_cache = {}
        self.teams_info = {}
        self.season_file_mapping = {}

        # Check for datasets subdirectory
        datasets_dir = os.path.join(self.data_dir, 'datasets')
        if os.path.exists(datasets_dir):
            self.data_dir = datasets_dir

        # Initialize NBA teams
        self._load_nba_teams()

        # Build metadata on startup with season verification
        self._build_metadata()

    def _load_nba_teams(self):
        """Load NBA teams from API or fallback"""
        if NBA_API_AVAILABLE:
            try:
                nba_teams = teams.get_teams()
                self.teams_info = {team['abbreviation']: team for team in nba_teams}
                self.teams_by_id = {team['id']: team for team in nba_teams}
                return
            except Exception as e:
                print(f"Error loading teams from NBA API: {e}")

        # Fallback teams
        print("Using fallback team data.")
        fallback_teams = [
            {'id': 1610612737, 'full_name': 'Atlanta Hawks', 'abbreviation': 'ATL'},
            {'id': 1610612738, 'full_name': 'Boston Celtics', 'abbreviation': 'BOS'},
            {'id': 1610612751, 'full_name': 'Brooklyn Nets', 'abbreviation': 'BKN'},
            {'id': 1610612766, 'full_name': 'Charlotte Hornets', 'abbreviation': 'CHA'},
            {'id': 1610612741, 'full_name': 'Chicago Bulls', 'abbreviation': 'CHI'},
            {'id': 1610612739, 'full_name': 'Cleveland Cavaliers', 'abbreviation': 'CLE'},
            {'id': 1610612742, 'full_name': 'Dallas Mavericks', 'abbreviation': 'DAL'},
            {'id': 1610612743, 'full_name': 'Denver Nuggets', 'abbreviation': 'DEN'},
            {'id': 1610612765, 'full_name': 'Detroit Pistons', 'abbreviation': 'DET'},
            {'id': 1610612744, 'full_name': 'Golden State Warriors', 'abbreviation': 'GSW'},
            {'id': 1610612745, 'full_name': 'Houston Rockets', 'abbreviation': 'HOU'},
            {'id': 1610612754, 'full_name': 'Indiana Pacers', 'abbreviation': 'IND'},
            {'id': 1610612746, 'full_name': 'LA Clippers', 'abbreviation': 'LAC'},
            {'id': 1610612747, 'full_name': 'Los Angeles Lakers', 'abbreviation': 'LAL'},
            {'id': 1610612763, 'full_name': 'Memphis Grizzlies', 'abbreviation': 'MEM'},
            {'id': 1610612748, 'full_name': 'Miami Heat', 'abbreviation': 'MIA'},
            {'id': 1610612749, 'full_name': 'Milwaukee Bucks', 'abbreviation': 'MIL'},
            {'id': 1610612750, 'full_name': 'Minnesota Timberwolves', 'abbreviation': 'MIN'},
            {'id': 1610612740, 'full_name': 'New Orleans Pelicans', 'abbreviation': 'NOP'},
            {'id': 1610612752, 'full_name': 'New York Knicks', 'abbreviation': 'NYK'},
            {'id': 1610612760, 'full_name': 'Oklahoma City Thunder', 'abbreviation': 'OKC'},
            {'id': 1610612753, 'full_name': 'Orlando Magic', 'abbreviation': 'ORL'},
            {'id': 1610612755, 'full_name': 'Philadelphia 76ers', 'abbreviation': 'PHI'},
            {'id': 1610612756, 'full_name': 'Phoenix Suns', 'abbreviation': 'PHX'},
            {'id': 1610612757, 'full_name': 'Portland Trail Blazers', 'abbreviation': 'POR'},
            {'id': 1610612758, 'full_name': 'Sacramento Kings', 'abbreviation': 'SAC'},
            {'id': 1610612759, 'full_name': 'San Antonio Spurs', 'abbreviation': 'SAS'},
            {'id': 1610612761, 'full_name': 'Toronto Raptors', 'abbreviation': 'TOR'},
            {'id': 1610612762, 'full_name': 'Utah Jazz', 'abbreviation': 'UTA'},
            {'id': 1610612764, 'full_name': 'Washington Wizards', 'abbreviation': 'WAS'}
        ]

        self.teams_info = {team['abbreviation']: team for team in fallback_teams}
        self.teams_by_id = {team['id']: team for team in fallback_teams}

    def _build_metadata(self):
        """Build metadata with correct season mapping by checking actual file contents"""
        if not os.path.exists(self.data_dir):
            print(f"Data directory not found: {self.data_dir}")
            return

        try:
            files = os.listdir(self.data_dir)
            csv_files = [f for f in files if f.endswith('.csv')]
        except Exception as e:
            print(f"Error accessing data directory: {e}")
            return

        # First pass: Identify what seasons each file actually contains
        for filename in csv_files:
            filepath = os.path.join(self.data_dir, filename)

            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 2:
                if 'po' in parts:
                    data_type = parts[0]
                    file_year = parts[-1]
                    is_playoff = True
                else:
                    data_type = parts[0]
                    file_year = parts[-1]
                    is_playoff = False

                if file_year.isdigit() and data_type == 'shotdetail':
                    actual_season = self._detect_season_from_file(filepath)
                    if actual_season:
                        self.season_file_mapping[f"{actual_season}_{data_type}_{'po' if is_playoff else 'reg'}"] = {
                            'filename': filename,
                            'path': filepath,
                            'type': data_type,
                            'playoff': is_playoff,
                            'actual_season': actual_season
                        }

        # Second pass: Build metadata cache using detected seasons
        for key, file_info in self.season_file_mapping.items():
            season = file_info['actual_season']
            data_type = file_info['type']
            is_playoff = file_info['playoff']

            if season not in self.metadata_cache:
                self.metadata_cache[season] = {}

            cache_key = f"{data_type}_{'po' if is_playoff else 'reg'}"
            self.metadata_cache[season][cache_key] = file_info

        # Third pass: Add non-shotdetail files using filename logic
        for filename in csv_files:
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 2:
                if 'po' in parts:
                    data_type = parts[0]
                    year = parts[-1]
                    is_playoff = True
                else:
                    data_type = parts[0]
                    year = parts[-1]
                    is_playoff = False

                if year.isdigit() and data_type != 'shotdetail':
                    year_int = int(year)
                    season = f"{year_int-1}-{year[2:]}"

                    if season not in self.metadata_cache:
                        self.metadata_cache[season] = {}

                    key = f"{data_type}_{'po' if is_playoff else 'reg'}"
                    if key not in self.metadata_cache[season]:
                        self.metadata_cache[season][key] = {
                            'filename': filename,
                            'path': os.path.join(self.data_dir, filename),
                            'type': data_type,
                            'playoff': is_playoff
                        }

        if not self.metadata_cache:
            print("No seasons found in metadata.")

    def _detect_season_from_file(self, filepath):
        """Detect actual season from file contents by checking game dates"""
        try:
            sample = pd.read_csv(filepath, nrows=1000)

            if 'GAME_DATE' not in sample.columns:
                return None

            dates = pd.to_datetime(sample['GAME_DATE'], format='%Y%m%d', errors='coerce')
            dates = dates.dropna()
            if len(dates) == 0:
                return None

            min_date = dates.min()
            max_date = dates.max()

            # Determine season based on date range
            # NBA seasons run from October to June of the following year
            if min_date.month >= 10:
                season_start_year = min_date.year
            else:
                season_start_year = min_date.year - 1

            season = f"{season_start_year}-{str(season_start_year + 1)[2:]}"

            expected_start = pd.Timestamp(f"{season_start_year}-10-01")
            expected_end = pd.Timestamp(f"{season_start_year + 1}-06-30")

            if min_date >= expected_start and max_date <= expected_end:
                return season
            else:
                # Keep as season even if partial
                return season

        except Exception as e:
            print(f"Error detecting season from {os.path.basename(filepath)}: {e}")
            return None

    def get_available_seasons(self):
        """Get available seasons from metadata"""
        return sorted(self.metadata_cache.keys(), reverse=True)

    def get_teams_for_season(self, season):
        """Get teams from NBA API (same for all seasons)"""
        return sorted(self.teams_info.keys())

    @lru_cache(maxsize=50)
    def get_players_for_team_season(self, season, team):
        """Get players from NBA API roster"""
        cache_key = f"{team}_{season}"

        if cache_key not in self.roster_cache:
            team_info = self.teams_info.get(team)
            if not team_info:
                print(f"Team {team} not found")
                return []

            if NBA_API_AVAILABLE:
                players_list = self._get_roster_from_api(team, season, team_info)
                if players_list:
                    self.roster_cache[cache_key] = players_list
                    return players_list

            # Fallback to shot data
            print(f"NBA API roster unavailable for {team}. Falling back to shot data.")
            players_list = self._get_players_from_shot_data(team, season, team_info)
            self.roster_cache[cache_key] = players_list

        return self.roster_cache.get(cache_key, [])

    def _get_roster_from_api(self, team, season, team_info):
        """Get roster from NBA API"""
        try:
            team_id = team_info['id']
            roster = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season
            )
            time.sleep(0.2)  # avoid rate limiting
            roster_df = roster.get_data_frames()[0]

            if not roster_df.empty:
                return roster_df['PLAYER'].tolist()
            else:
                return []

        except Exception as e:
            print(f"NBA API roster error for {team}: {e}")
            return []

    def _get_players_from_shot_data(self, team, season, team_info):
        """Fallback: Get players from shot data"""
        try:
            shot_file = self._get_file_path(season, 'shotdetail', playoff=False)
            if not shot_file:
                print(f"No shot data file for {season}")
                return []

            df = pd.read_csv(shot_file)
            df.columns = df.columns.str.upper()

            team_full_name = team_info['full_name']

            team_data = df[df['TEAM_NAME'] == team_full_name]

            if len(team_data) == 0:
                possible_names = [
                    team_full_name,
                    team_full_name.replace(' ', ''),
                    team_info['abbreviation']
                ]
                for name in possible_names:
                    team_data = df[df['TEAM_NAME'].str.contains(name, na=False, case=False)]
                    if len(team_data) > 0:
                        # keep one helpful message
                        print(f"Matched team name pattern: {name}")
                        break

            if len(team_data) > 0:
                players_list = sorted(team_data['PLAYER_NAME'].dropna().unique())
                players_list = [str(p).strip() for p in players_list if str(p) != 'nan' and str(p).strip()]
                return players_list
            else:
                print(f"No players found in shot data for {team}")
                return []

        except Exception as e:
            print(f"Error loading players from shot data: {e}")
            return []

    def load_player_shots(self, season, player_name, include_playoffs=True):
        """Load shots for specific player with correct season handling"""
        shots = []

        # Load regular season
        reg_file = self._get_file_path(season, 'shotdetail', playoff=False)
        if reg_file:
            reg_shots = self._load_player_from_file(reg_file, player_name)
            if len(reg_shots) > 0:
                reg_shots['season_type'] = 'Regular'
                shots.append(reg_shots)
        else:
            print(f"No regular season file found for {season}")

        # Load playoffs if requested
        if include_playoffs:
            po_file = self._get_file_path(season, 'shotdetail', playoff=True)
            if po_file:
                po_shots = self._load_player_from_file(po_file, player_name)
                if len(po_shots) > 0:
                    po_shots['season_type'] = 'Playoffs'
                    shots.append(po_shots)
            else:
                # informative but not noisy
                pass

        if shots:
            combined = pd.concat(shots, ignore_index=True)
            standardized = self._standardize_shot_data(combined)
            if 'game_date' in standardized.columns:
                try:
                    min_date = standardized['game_date'].min()
                    max_date = standardized['game_date'].max()
                    # Keep this as a debug aid but concise
                    print(f"Loaded {len(standardized)} shots for {player_name} [{min_date.date()} .. {max_date.date()}]")
                except Exception:
                    print(f"Loaded {len(standardized)} shots for {player_name}")
            return standardized
        else:
            print(f"No shots found for {player_name} in {season}")
            return pd.DataFrame()

    def _load_player_from_file(self, filepath, player_name):
        """Load specific player's shots from file"""
        try:
            chunk_size = 10000
            player_shots = []

            for chunk in pd.read_csv(filepath, chunksize=chunk_size):
                player_chunk = chunk[chunk['PLAYER_NAME'] == player_name]
                if len(player_chunk) > 0:
                    player_shots.append(player_chunk)

            if player_shots:
                return pd.concat(player_shots, ignore_index=True)
            else:
                return pd.DataFrame()

        except Exception as e:
            print(f"Error loading player from {filepath}: {e}")
            return pd.DataFrame()

    def _get_file_path(self, season, data_type, playoff=False):
        """Get file path for specific data type and season"""
        key = f"{data_type}_{'po' if playoff else 'reg'}"
        if season in self.metadata_cache and key in self.metadata_cache[season]:
            return self.metadata_cache[season][key]['path']
        return None

    def _standardize_shot_data(self, df):
        """Standardize column names and data types"""
        df = df.copy()
        df.columns = df.columns.str.lower()

        column_map = {
            'loc_x': 'x',
            'loc_y': 'y'
        }
        df = df.rename(columns=column_map)

        if 'game_date' in df.columns:
            try:
                df['game_date'] = pd.to_datetime(df['game_date'], format='%Y%m%d')
            except Exception:
                try:
                    df['game_date'] = pd.to_datetime(df['game_date'])
                except Exception:
                    print("Could not parse game_date column")

        return df

    def validate_data_availability(self, season):
        """Check what data types are available for a season"""
        available = {
            'shotdetail': bool(self._get_file_path(season, 'shotdetail', False)),
            'nbastats': bool(self._get_file_path(season, 'nbastats', False)),
            'datanba': bool(self._get_file_path(season, 'datanba', False)),
            'pbpstats': bool(self._get_file_path(season, 'pbpstats', False)),
            'matchups': bool(self._get_file_path(season, 'matchups', False))
        }
        return available

    def debug_season_mapping(self):
        """Debug method to show season-to-file mapping (silenced by default)"""
        pass

    def verify_season_data(self, season):
        """Verify that the correct season data is being loaded"""
        shot_file = self._get_file_path(season, 'shotdetail', False)
        if shot_file:
            try:
                sample = pd.read_csv(shot_file, nrows=100)
                dates = pd.to_datetime(sample['GAME_DATE'], format='%Y%m%d')

                season_start_year = int(season.split('-')[0])
                expected_start = pd.Timestamp(f"{season_start_year}-10-01")
                expected_end = pd.Timestamp(f"{season_start_year + 1}-06-30")

                in_range = ((dates >= expected_start) & (dates <= expected_end)).sum()
                total = len(dates)

                ratio = (in_range / total * 100) if total else 0
                print(f"Verification for {season}: {os.path.basename(shot_file)} | Match {in_range}/{total} ({ratio:.1f}%)")

            except Exception as e:
                print(f"Error verifying season {season}: {e}")
        else:
            print(f"No file found for {season}")


# Example usage and testing (minimal console output)
if __name__ == "__main__":
    print("Testing EXE-Ready NBA Data Manager...")

    dm = NBADataManager()

    # Show a quick verification if seasons exist
    seasons = dm.get_available_seasons()
    if seasons:
        test_season = seasons[0]
        dm.verify_season_data(test_season)

        teams_list = dm.get_teams_for_season(test_season)
        print(f"Teams available: {len(teams_list)}")

        if teams_list:
            test_team = teams_list[0]
            team_players = dm.get_players_for_team_season(test_season, test_team)
            print(f"Players for {test_team}: {len(team_players)}")

            if team_players:
                test_player = team_players[0]
                shots = dm.load_player_shots(test_season, test_player)
                print(f"Shots for {test_player}: {len(shots)}")

    print("NBA Data Manager test complete.")
