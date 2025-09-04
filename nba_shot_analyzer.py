import sys
import os
import subprocess
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from typing import Union, Dict
from PyQt6.QtWidgets import (QApplication, QMainWindow, QMessageBox, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QFileDialog, QWidget, QLabel,
                             QDialog, QComboBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QShortcut, QKeySequence, QPixmap
from mainwindow import Ui_MainWindow
from nba_data_manager import NBADataManager, EnhancedNBADataManager
from nba_filter_engine import NBAFilterEngine

try:
    from nba_api_helper import get_nba_teams
except ImportError:
    def get_nba_teams():
        return {
            'abbreviations': ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 
                            'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA',
                            'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX',
                            'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'],
            'by_abbreviation': {}
        }

class SmartResolutionHeatmapGenerator(QThread):
    """Smart heatmap generator that creates the right resolution for display"""
    
    image_ready = pyqtSignal(str)
    
    def __init__(self, shot_data, player_name, zones_data, frame_size):
        super().__init__()
        self.shot_data = shot_data
        self.player_name = player_name
        self.zones_data = zones_data
        self.frame_width = frame_size.width()
        self.frame_height = frame_size.height()
    
    def run(self):
        
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            image_path = temp_file.name
            temp_file.close()
            
            self.create_smart_resolution_heatmap(image_path)
            self.image_ready.emit(image_path)
            
        except Exception as e:
            print(f"Error generating smart heatmap: {e}")
    
    def create_smart_resolution_heatmap(self, output_path):
        """Create heatmap with PERFECT resolution for your frame"""
        import matplotlib
        matplotlib.use('Agg')
        
        # Calculate PERFECT figure size for your frame (521x431 pixels from your UI)
        frame_aspect = self.frame_width / self.frame_height  # ~1.21
        
        # Set figure size to match your frame EXACTLY
        fig_width_inches = 10.4  # Good size for quality
        fig_height_inches = fig_width_inches / frame_aspect
        
        print(f"🎯 Creating smart heatmap for frame {self.frame_width}x{self.frame_height}")
        print(f"📐 Using figure size: {fig_width_inches:.1f}x{fig_height_inches:.1f} inches")
        
        # Create figure with PERFECT aspect ratio
        fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#1d428a')
        
        # Draw court with crisp details
        self.draw_crisp_court(ax)
        
        # Add heatmap data
        if self.shot_data is not None and not self.shot_data.empty:
            self.add_crisp_heatmap(ax)
            self.add_crisp_zone_labels(ax)
        
        # Perfect court dimensions
        ax.set_xlim(-260, 260)
        ax.set_ylim(-50, 430)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Title sized perfectly for frame
        plt.title(f'{self.player_name} - Shot Analysis', 
                 color='white', fontsize=16, fontweight='bold', pad=15)
        
        # Add legend with perfect sizing
        self.add_perfect_legend(ax)
        
        # Save with SMART DPI - high enough for quality, not wasteful
        target_dpi = max(150, min(300, self.frame_width / fig_width_inches))
        print(f"🎯 Using smart DPI: {target_dpi:.0f}")
        
        plt.savefig(output_path, 
           dpi=target_dpi,
           bbox_inches='tight', 
           facecolor='#2b2b2b',
           edgecolor='none',
           transparent=False,
           pad_inches=0.1,
           quality=95)
        plt.close()
        
        print(f"✅ Smart resolution heatmap created: {target_dpi:.0f} DPI")
    
    def draw_crisp_court(self, ax):
        """Draw court with crisp, visible lines"""
        line_color = 'white'
        line_width = 2.5  # Optimized for your frame size
        
        # Court outline
        court = patches.Rectangle((-250, -47.5), 500, 470, 
                                linewidth=line_width, edgecolor=line_color, 
                                facecolor='none')
        ax.add_patch(court)
        
        # Center circle
        center_circle = patches.Circle((0, 0), 60, linewidth=line_width, 
                                     edgecolor=line_color, facecolor='none')
        ax.add_patch(center_circle)
        
        # 3-point arc
        three_point_arc = patches.Arc((0, 0), 2*237.5, 2*237.5, 
                                    theta1=22, theta2=158, linewidth=line_width, 
                                    edgecolor=line_color)
        ax.add_patch(three_point_arc)
        
        # 3-point corners
        ax.plot([-220, -220], [-47.5, 92.5], color=line_color, linewidth=line_width)
        ax.plot([220, 220], [-47.5, 92.5], color=line_color, linewidth=line_width)
        
        # Paint area
        paint = patches.Rectangle((-80, -47.5), 160, 190, 
                                linewidth=line_width, edgecolor=line_color, 
                                facecolor='none')
        ax.add_patch(paint)
        
        # Free throw circle
        ft_semicircle = patches.Arc((0, 142.5), 120, 120,  # diameter = 120 (radius 60)
                           theta1=0, theta2=180,    # 0° to 180° = top semicircle
                           linewidth=line_width, 
                           edgecolor=line_color, facecolor='none', alpha=0.9)
        ax.add_patch(ft_semicircle)
        
        # Basket - perfect size
        basket = patches.Circle((0, 0), 7, linewidth=4, 
                              edgecolor='orange', facecolor='orange')
        ax.add_patch(basket)
        
        # Backboard
        ax.plot([-30, 30], [-7.5, -7.5], color='white', linewidth=5)
    
    def add_crisp_heatmap(self, ax):
        """Add perfectly crisp heatmap for your frame size"""
        if 'x' not in self.shot_data.columns or 'y' not in self.shot_data.columns:
            return
        
        x = self.shot_data['x'].values
        y = self.shot_data['y'].values
        made = self.shot_data['shot_made_flag'].values
        
        # Optimized color scheme for visibility
        colors_hot = ['#001a4d', '#003380', '#0066cc', '#0099ff', '#33ccff', 
                      '#66ffcc', '#ccff66', '#ffcc33', '#ff9900', '#ff3300']
        cmap_hot = LinearSegmentedColormap.from_list('crisp_hot', colors_hot)
        
        made_x = x[made == 1]
        made_y = y[made == 1]
        
        if len(made_x) > 0:
            # SMART bin count based on frame size and data density
            shot_density = len(made_x) / 1000  # shots per 1000
            optimal_bins = max(30, min(60, int(40 + shot_density * 20)))
            
            print(f"🎯 Using {optimal_bins} bins for {len(made_x)} made shots")
            
            H_made, xedges, yedges = np.histogram2d(made_x, made_y, 
                                                   bins=optimal_bins,
                                                   range=[[-250, 250], [-47.5, 422.5]])
            
            # Smart smoothing
            sigma = 0.1
            H_made = ndimage.gaussian_filter(H_made, sigma=sigma)
            
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            ax.imshow(H_made.T, extent=extent, origin='lower', 
                      cmap=cmap_hot, alpha=0.8, aspect='auto',
                      interpolation='none')
        
        # Shot dots with perfect size for frame
        dot_size = max(6, min(12, self.frame_width / 50))
        edge_width = max(0.5, min(1.0, dot_size / 10))
        
        ax.scatter(made_x, made_y, c='#4CAF50', s=dot_size, alpha=0.9,
                  edgecolors='white', linewidth=edge_width, label='Made')
        
        missed_x = x[made == 0]
        missed_y = y[made == 0]
        ax.scatter(missed_x, missed_y, c='#F44336', s=dot_size, alpha=0.7,
                  edgecolors='white', linewidth=edge_width, label='Missed')
    
    def add_crisp_zone_labels(self, ax):
        """Add zone labels with perfect sizing"""
        if not self.zones_data:
            return
        
        zone_positions = {
            'Restricted Area': (0, 60),
            'In The Paint (Non-RA)': (0, 130),
            'Mid-Range': (0, 200),
            'Above the Break 3': (0, 300),
            'Left Corner 3': (-200, 40),
            'Right Corner 3': (200, 40),
        }
        
        # Font size based on frame size
        label_font_size = max(8, min(14, self.frame_width / 40))
        
        for zone_name, stats in self.zones_data.items():
            if zone_name in zone_positions:
                x, y = zone_positions[zone_name]
                attempts = stats.get('attempted', 0)
                made = stats.get('made', 0)
                pct = stats.get('percentage', 0)
                
                if attempts > 0:
                    label_text = f"{pct:.1f}%\n{made}/{attempts}"
                    
                    # Color based on efficiency
                    if pct >= 50:
                        bg_color = '#4CAF50'
                    elif pct >= 40:
                        bg_color = '#FFC107'
                    elif pct >= 30:
                        bg_color = '#FF9800'
                    else:
                        bg_color = '#F44336'
                    
                    ax.text(x, y, label_text, fontsize=label_font_size, fontweight='bold',
                           ha='center', va='center', color='white',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor=bg_color,
                                   edgecolor='white', linewidth=1.5, alpha=0.95))
    
    def add_perfect_legend(self, ax):
        """Add legend with perfect sizing for frame"""
        marker_size = max(8, min(15, self.frame_width / 35))
        font_size = max(8, min(12, self.frame_width / 45))
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50', 
                      markersize=marker_size, label='Made Shots', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', 
                      markersize=marker_size, label='Missed Shots', linestyle='None'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(0.98, 0.98), fancybox=True, 
                 shadow=True, frameon=True, facecolor='white', 
                 edgecolor='gray', fontsize=font_size)

class ShotZoneCalculator:
    """Calculate shooting zones using original NBA zones"""
    
    def __init__(self):
        self.zone_stats: Dict[str, Dict[str, Union[int, float]]] = {}
        self.zone_column = None
    
    def calculate_zones(self, shot_data: pd.DataFrame) -> Dict[str, Dict[str, Union[int, float]]]:
        """Use original NBA zones directly"""
        if shot_data.empty:
            return {}
        
        print(f"🎯 Calculating zones for {len(shot_data)} shots")
        
        zone_column = self._find_best_zone_column(shot_data)
        
        if not zone_column:
            print("❌ No zone column found in data")
            return {}
        
        print(f"📊 Using original NBA zones from: {zone_column}")
        self.zone_column = zone_column
        
        data = shot_data.copy()
        if 'Backcourt' in data[zone_column].values:
            original_count = len(data)
            data = data[data[zone_column] != 'Backcourt']
            filtered_count = len(data)
            print(f"🗑️ Filtered out {original_count - filtered_count} backcourt shots")
        
        zone_stats = data.groupby(zone_column).agg({
            'shot_made_flag': ['count', 'sum']
        })
        
        zone_stats.columns = ['attempted', 'made']
        zone_stats['percentage'] = (zone_stats['made'] / zone_stats['attempted'] * 100).round(1)
        
        result = {}
        for zone_name, row in zone_stats.iterrows():
            if pd.notna(zone_name):
                result[str(zone_name)] = {
                    'attempted': int(row['attempted']),
                    'made': int(row['made']),
                    'percentage': float(row['percentage'])
                }
        
        self.zone_stats = result
        
        print(f"✅ Using original NBA zones ({len(result)} zones):")
        for zone_name, stats in result.items():
            print(f"   {zone_name}: {stats['percentage']:.1f}% ({stats['made']}/{stats['attempted']})")
        
        return result
    
    def _find_best_zone_column(self, shot_data: pd.DataFrame) -> Union[str, None]:
        """Find the best zone column to use for analysis"""
        zone_columns = [
            'shot_zone_basic',
            'shot_zone_area',
            'shot_zone_range',
            'action_type',
            'shot_type'
        ]
        
        for col in zone_columns:
            if col in shot_data.columns:
                unique_zones = shot_data[col].nunique()
                print(f"   {col}: {unique_zones} unique zones")
                
                if unique_zones > 1:
                    return col
        
        return None
    
    def get_zone_summary(self) -> Dict[str, Union[int, float]]:
        """Get summary statistics"""
        if not self.zone_stats:
            return {'total_attempted': 0, 'total_made': 0, 'overall_percentage': 0}
        
        total_attempted = sum(zone['attempted'] for zone in self.zone_stats.values())
        total_made = sum(zone['made'] for zone in self.zone_stats.values())
        overall_pct = (total_made / total_attempted * 100) if total_attempted > 0 else 0
        
        return {
            'total_attempted': total_attempted,
            'total_made': total_made,
            'overall_percentage': round(overall_pct, 1)
        }


class Optimal700x550HeatmapGenerator(QThread):
    """OPTIMAL: Perfect quality heatmap for 700x550 frame"""
    
    image_ready = pyqtSignal(str)
    
    def __init__(self, shot_data, player_name, zones_data, frame_size):
        super().__init__()
        self.shot_data = shot_data
        self.player_name = player_name
        self.zones_data = zones_data
        self.frame_width = frame_size.width()   # 700
        self.frame_height = frame_size.height() # 550
    
    def run(self):
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            image_path = temp_file.name
            temp_file.close()
            
            print(f"🎨 Creating OPTIMAL heatmap for {self.frame_width}x{self.frame_height}")
            self.create_optimal_heatmap(image_path)
            self.image_ready.emit(image_path)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    def draw_optimal_court(self, ax, target_width):
        """Draw optimal court with perfect scaling"""
        # Scale line width based on target resolution
        line_width = max(2.0, target_width / 600)  # Scales with resolution
        line_color = '#ffffff'
        
        print(f"🏀 Drawing court with line width: {line_width:.1f}")
        
        # Court outline
        court = patches.Rectangle((-250, -47.5), 500, 470, 
                                linewidth=line_width, edgecolor=line_color, 
                                facecolor='none', alpha=0.9)
        ax.add_patch(court)
        
        # Center circle
        center_circle = patches.Circle((0, 0), 60, linewidth=line_width, 
                                    edgecolor=line_color, facecolor='none', alpha=0.9)
        ax.add_patch(center_circle)
        
        # 3-point arc
        three_point_arc = patches.Arc((0, 0), 2*237.5, 2*237.5, 
                                    theta1=22, theta2=158, linewidth=line_width, 
                                    edgecolor=line_color, alpha=0.9)
        ax.add_patch(three_point_arc)
        
        # 3-point corners
        ax.plot([-220, -220], [-47.5, 92.5], color=line_color, linewidth=line_width, alpha=0.9)
        ax.plot([220, 220], [-47.5, 92.5], color=line_color, linewidth=line_width, alpha=0.9)
        
        # Paint area
        paint = patches.Rectangle((-80, -47.5), 160, 190, 
                                linewidth=line_width, edgecolor=line_color, 
                                facecolor='none', alpha=0.9)
        ax.add_patch(paint)
        
        # Free throw circle
        ft_semicircle = patches.Arc((0, 142.5), 120, 120,  # diameter = 120 (radius 60)
                           theta1=0, theta2=180,    # 0° to 180° = top semicircle
                           linewidth=line_width, 
                           edgecolor=line_color, facecolor='none', alpha=0.9)
        ax.add_patch(ft_semicircle)
        
        # Basketball rim - scaled properly
        rim_size = max(7, target_width / 200)
        rim = patches.Circle((0, 0), rim_size, linewidth=line_width, 
                        edgecolor='#ff6b35', facecolor='#ff6b35', alpha=1.0)
        ax.add_patch(rim)
        
        # Backboard
        backboard_width = line_width * 1.5
        ax.plot([-30, 30], [-7.5, -7.5], color=line_color, linewidth=backboard_width, alpha=0.9)
    
    def create_optimal_heatmap(self, output_path):
        """Create optimal quality heatmap for 700x550"""
        import matplotlib
        matplotlib.use('Agg')
        
        # Create at 2.5x resolution for excellent quality when scaled down
        multiplier = 2.5
        target_width = int(self.frame_width * multiplier)   # 1750
        target_height = int(self.frame_height * multiplier) # 1375
        
        # Optimal DPI
        dpi = 175
        fig_width = target_width / dpi
        fig_height = target_height / dpi
        
        print(f"🎯 Creating {target_width}x{target_height} for perfect {self.frame_width}x{self.frame_height} display")
        print(f"📐 Figure: {fig_width:.1f}x{fig_height:.1f} inches at {dpi} DPI")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#0a1929')
        
        # Perfect margins
        plt.subplots_adjust(left=0.015, right=0.985, top=0.935, bottom=0.065)
        
        # Draw court
        self.draw_optimal_court(ax, target_width)
        
        # Add heatmap
        if self.shot_data is not None and not self.shot_data.empty:
            self.add_optimal_heatmap(ax, target_width)
            self.add_optimal_zones(ax, target_width)
        
        # Court dimensions
        ax.set_xlim(-260, 260)
        ax.set_ylim(-50, 430)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Perfect title
        title_size = target_width / 62  # 28px - scales down to 11px
        plt.title(f'{self.player_name} - Shot Analysis', 
                 color='white', fontsize=title_size, fontweight='bold', pad=15)
        
        # Add legend
        self.add_optimal_legend(ax, target_width)
        
        print(f"💾 Saving optimal resolution...")
        
        # Save
        plt.savefig(output_path, 
                   dpi=dpi,
                   bbox_inches='tight',
                   pad_inches=0.02,
                   facecolor='#1a1a1a',
                   edgecolor='none',
                   transparent=False)
        plt.close()
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            test_pixmap = QPixmap(output_path)
            if not test_pixmap.isNull():
                actual_w, actual_h = test_pixmap.width(), test_pixmap.height()
                print(f"✅ OPTIMAL: {actual_w}x{actual_h} pixels ({file_size:,} bytes)")
                scale_factor = actual_w / self.frame_width
                print(f"📏 Scale down factor: {scale_factor:.1f}x for crisp display")
    
    def draw_export_court(self, ax):
        """Draw professional NBA court for export"""
        # Professional colors
        line_color = '#ffffff'  # Dark professional blue
        line_width = 2.5
        
        # Court outline
        court = patches.Rectangle((-250, -47.5), 500, 470, 
                                linewidth=line_width, edgecolor=line_color, 
                                facecolor='none', alpha=0.8)
        ax.add_patch(court)
        
        # Center circle
        center_circle = patches.Circle((0, 0), 60, linewidth=line_width, 
                                    edgecolor=line_color, facecolor='none', alpha=0.8)
        ax.add_patch(center_circle)
        
        # 3-point arc
        three_point_arc = patches.Arc((0, 0), 2*237.5, 2*237.5, 
                                    theta1=22, theta2=158, linewidth=line_width, 
                                    edgecolor=line_color, alpha=0.8)
        ax.add_patch(three_point_arc)
        
        # 3-point corners
        ax.plot([-220, -220], [-47.5, 92.5], color=line_color, linewidth=line_width, alpha=0.8)
        ax.plot([220, 220], [-47.5, 92.5], color=line_color, linewidth=line_width, alpha=0.8)
        
        # Paint area
        paint = patches.Rectangle((-80, -47.5), 160, 190, 
                                linewidth=line_width, edgecolor=line_color, 
                                facecolor='none', alpha=0.8)
        ax.add_patch(paint)
        
        # Free throw circle
        ft_semicircle = patches.Arc((0, 142.5), 120, 120,  # diameter = 120 (radius 60)
                           theta1=0, theta2=180,    # 0° to 180° = top semicircle
                           linewidth=line_width, 
                           edgecolor=line_color, facecolor='none', alpha=0.9)
        ax.add_patch(ft_semicircle)
        
        # Professional basketball rim
        rim = patches.Circle((0, 0), 9, linewidth=line_width, 
                        edgecolor='#e74c3c', facecolor='#e74c3c', alpha=0.9)
        ax.add_patch(rim)
        
        # Backboard
        ax.plot([-30, 30], [-7.5, -7.5], color=line_color, linewidth=line_width * 1.2, alpha=0.8)
    
    def add_optimal_heatmap(self, ax, target_width):
        """Add optimal heatmap with perfect resolution scaling"""
        if 'x' not in self.shot_data.columns or 'y' not in self.shot_data.columns:
            return
        
        x = self.shot_data['x'].values
        y = self.shot_data['y'].values
        made = self.shot_data['shot_made_flag'].values
        
        # High-quality color scheme
        colors_hot = [
            '#001122', '#003366', '#0066aa', '#0099ee', '#33ccff', 
            '#66ffcc', '#ccff66', '#ffcc33', '#ff9900', '#ff4400', '#cc0000'
        ]
        cmap_hot = LinearSegmentedColormap.from_list('optimal_hot', colors_hot)
        
        made_x = x[made == 1]
        made_y = y[made == 1]
        
        if len(made_x) > 0:
            # Scale bin count with target resolution
            base_bins = 45
            resolution_factor = target_width / 1000
            optimal_bins = int(base_bins * resolution_factor)
            optimal_bins = max(30, min(80, optimal_bins))
            
            print(f"🎯 Using {optimal_bins} bins for {len(made_x)} made shots at {target_width}px width")
            
            H_made, xedges, yedges = np.histogram2d(made_x, made_y, 
                                                   bins=optimal_bins,
                                                   range=[[-250, 250], [-47.5, 422.5]])
            
            # Optimal smoothing based on resolution
            sigma = 0.8 * (target_width / 1000)
            H_made = ndimage.gaussian_filter(H_made, sigma=sigma)
            
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            ax.imshow(H_made.T, extent=extent, origin='lower', 
                      cmap=cmap_hot, alpha=0.75, aspect='auto',
                      interpolation='bilinear')
        
        # Scale shot dots with resolution
        dot_size = max(8, target_width / 140)
        edge_width = max(0.8, target_width / 1400)
        
        print(f"🎯 Shot dots: size={dot_size:.1f}, edge={edge_width:.1f}")
        
        # Made shots - vibrant green
        ax.scatter(made_x, made_y, c='#00FF7F', s=dot_size, alpha=0.9,
                  edgecolors='white', linewidth=edge_width, label='Made')
        
        # Missed shots - vibrant red
        missed_x = x[made == 0]
        missed_y = y[made == 0]
        ax.scatter(missed_x, missed_y, c='#FF4444', s=dot_size * 0.85, alpha=0.8,
                  edgecolors='white', linewidth=edge_width, label='Missed')
    
    def add_optimal_zones(self, ax, target_width):
        """Add zone labels with optimal scaling"""
        if not self.zones_data:
            return
        
        zone_positions = {
            'Restricted Area': (0, 70),
            'In The Paint (Non-RA)': (0, 140),
            'Mid-Range': (0, 210),
            'Above the Break 3': (0, 305),
            'Left Corner 3': (-180, 50),
            'Right Corner 3': (180, 50),
        }
        
        # Scale font size with resolution
        label_font_size = max(10, target_width / 120)
        
        print(f"🎯 Zone labels: font size={label_font_size:.1f}")
        
        for zone_name, stats in self.zones_data.items():
            if zone_name in zone_positions:
                x, y = zone_positions[zone_name]
                attempts = stats.get('attempted', 0)
                made = stats.get('made', 0)
                pct = stats.get('percentage', 0)
                
                if attempts > 0:
                    label_text = f"{pct:.1f}%\n{made}/{attempts}"
                    
                    # Color based on efficiency
                    if pct >= 50:
                        bg_color = '#00C853'  # Bright green
                    elif pct >= 40:
                        bg_color = '#FFB300'  # Bright orange
                    elif pct >= 30:
                        bg_color = '#FF8F00'  # Dark orange
                    else:
                        bg_color = '#D32F2F'  # Bright red
                    
                    # Scale padding with resolution
                    padding = max(0.4, target_width / 3000)
                    
                    ax.text(x, y, label_text, fontsize=label_font_size, fontweight='bold',
                           ha='center', va='center', color='white',
                           bbox=dict(boxstyle=f'round,pad={padding}', facecolor=bg_color,
                                   edgecolor='white', linewidth=2, alpha=0.95))
    
    def add_optimal_legend(self, ax, target_width):
        """Add perfectly sized legend for optimal quality"""
        marker_size = max(10, target_width / 140)  # Perfect marker size
        font_size = max(9, target_width / 150)     # Perfect font size
        
        print(f"🎯 Optimal legend: marker={marker_size:.1f}, font={font_size:.1f}")
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00FF7F', 
                      markersize=marker_size, label='Made Shots', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4444', 
                      markersize=marker_size, label='Missed Shots', linestyle='None'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(0.98, 0.98), fancybox=True, 
                 shadow=True, frameon=True, facecolor='white', 
                 edgecolor='gray', fontsize=font_size, framealpha=1.0)


# Enhanced update method for your main class
def update_court_image_ultra_quality(self, image_path):
    """ULTRA QUALITY VERSION: Update the court image with maximum quality rendering"""
    try:
        print(f"🖼️ [ULTRA QUALITY] Loading heatmap from: {image_path}")
        
        # Verify file exists and has content
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            self.show_error_message("Heatmap file not found")
            return
        
        # Check file size
        file_size = os.path.getsize(image_path)
        print(f"📁 Image file size: {file_size:,} bytes")
        
        if file_size < 1000:
            print("❌ Image file too small, likely corrupted")
            self.show_error_message("Heatmap generation failed")
            return
        
        # Load the image
        pixmap = QPixmap(image_path)
        
        if pixmap.isNull():
            print(f"❌ Failed to load pixmap from: {image_path}")
            self.show_error_message("Failed to load heatmap image")
            return
        
        print(f"✅ Pixmap loaded successfully. Original size: {pixmap.width()}x{pixmap.height()}")
        
        # Get frame dimensions
        frame_width = self.ui.frame.width()
        frame_height = self.ui.frame.height()
        print(f"📏 Frame dimensions: {frame_width}x{frame_height}")
        
        if frame_width <= 0 or frame_height <= 0:
            print("❌ Invalid frame dimensions")
            return
        
        # Clear any existing content in the frame
        for child in self.ui.frame.findChildren(QLabel):
            child.deleteLater()
        
        # Force processing of the deleteLater() calls
        QApplication.processEvents()
        
        # Create new label for the heatmap with explicit parent
        heatmap_label = QLabel(self.ui.frame)
        heatmap_label.setObjectName("ultra_quality_heatmap_display_label")
        
        # Set the label to fill the entire frame
        heatmap_label.setGeometry(0, 0, frame_width, frame_height)
        
        # Scale the image with ULTRA HIGH QUALITY transformation
        scaled_pixmap = pixmap.scaled(
            frame_width, frame_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation  # Highest quality scaling
        )
        
        print(f"📏 Scaled image size: {scaled_pixmap.width()}x{scaled_pixmap.height()}")
        
        # Set the scaled pixmap to the label
        heatmap_label.setPixmap(scaled_pixmap)
        heatmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Enhanced styling for maximum visual quality
        heatmap_label.setStyleSheet("""
            QLabel {
                border: none;
                background-color: transparent;
                padding: 0px;
                margin: 0px;
            }
        """)
        
        # Make sure the label is visible and on top
        heatmap_label.show()
        heatmap_label.raise_()
        
        # Force updates with maximum refresh
        self.ui.frame.update()
        self.ui.centralwidget.update()
        self.update()
        
        # Process all pending events
        QApplication.processEvents()
        
        print("✅ [ULTRA QUALITY] Ultra high quality heatmap successfully displayed!")
        
        # Store reference to prevent garbage collection
        self.current_heatmap_label = heatmap_label
        
        # Delayed cleanup of temporary file
        QApplication.instance().processEvents()
        import time
        time.sleep(0.5)
        
        try:
            if os.path.exists(image_path):
                os.unlink(image_path)
                print("🗑️ Temporary heatmap file cleaned up")
        except Exception as cleanup_error:
            print(f"⚠️ Could not clean up temp file: {cleanup_error}")
            
    except Exception as e:
        print(f"❌ [ULTRA QUALITY] Error updating heatmap display: {e}")
        import traceback
        traceback.print_exc()
        self.show_error_message(f"Display error: {str(e)}")

class NBAShotAnalyzer(QMainWindow):
    """NBA Shot Analyzer with Smart Resolution Heatmap System"""
    
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.statusbar.setVisible(True)
        self.ui.statusbar.show()
        self.ui.statusbar.setFixedHeight(45)
        self.ui.statusbar.setStyleSheet("""
        QStatusBar {
            background-color: transparent;
            border: none;
            padding-top: 2px;
            padding-bottom: 8px;
            padding-left: 5px;
            font-size: 11px;
            color: white;
            font-weight: bold;
        }
        QStatusBar::item {
            border: none;
        }
    """)
        # Initialize components
        self.data_manager = EnhancedNBADataManager()
        self.filter_engine = None
        self.zone_calculator = ShotZoneCalculator()
        
        self.shot_data = None
        self.current_season = None
        self.current_team = None
        self.current_player = None
        self.current_heatmap_label = None
        
        # Setup and connect
        self.setup_ui()
        self.setup_dropdown_hover()
        self.connect_signals()
        self.setup_debug_shortcuts()
        self.setup_export_functionality()
        
    def setup_ui(self):
        """Setup UI using your actual ComboBox names"""
        
        # Load seasons
        seasons = self.data_manager.get_available_seasons()
        self.ui.comboBox_9.clear()
        self.ui.comboBox_9.addItems(["Select Season"] + seasons)
        
        # Setup other dropdowns
        self.ui.comboBox_10.clear()
        self.ui.comboBox_10.addItems(["Select Team"])
        self.ui.comboBox_10.setEnabled(False)
        
        self.ui.comboBox_11.clear()
        self.ui.comboBox_11.addItems(["Select Player"])
        self.ui.comboBox_11.setEnabled(False)
        
        # Setup filters
        self.setup_filters()
        self.setup_labels()
        
        # Initially disable calculate
        self.ui.pushButton.setEnabled(False)
        
        print(f"✅ Optimized resolution UI setup complete with {len(seasons)} seasons")
        
    def setup_filters(self):
        """Setup filter dropdowns with enhanced options"""
        # Left side filters
        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(['All', 'Home', 'Away'])
        
        self.ui.comboBox_2.clear()
        self.ui.comboBox_2.addItems(['All', '1st', '2nd', '3rd', '4th', 'OT'])
        
        # Enhanced Season Phase filter
        self.ui.comboBox_3.clear()
        self.ui.comboBox_3.addItems([
            'All', 
            'Regular Season Only', 
            'Early Season (1-25)', 
            'Mid Season (26-60)', 
            'Late Season (61-82)', 
            'Playoffs Only'
        ])
        
        self.ui.comboBox_4.clear()
        self.ui.comboBox_4.addItems(['All', 'Clutch (±5)', 'Close (±10)', 'Blowout (>10)'])
        
        # Right side filters
        self.ui.comboBox_5.clear()
        self.ui.comboBox_5.addItems(['All', 'Leading', 'Trailing', 'Tied'])
        
        # Fixed Rest Days filter (logical order)
        self.ui.comboBox_6.clear()
        self.ui.comboBox_6.addItems([
            'All', 
            'Back-to-Back (0 days)', 
            '1 Day Rest', 
            '2 Days Rest', 
            '3+ Days Rest'
        ])
        
        # Enhanced Streak filter (removed short streaks option)
        self.ui.comboBox_7.clear()
        self.ui.comboBox_7.addItems([
            'All',
            'Win Streak (Any)',
            '2+ Game Win Streak',
            '3+ Game Win Streak', 
            '5+ Game Win Streak',
            'Loss Streak (Any)',
            '2+ Game Loss Streak',
            '3+ Game Loss Streak',
            '5+ Game Loss Streak',
            'No Streak'
        ])
        
        # Enhanced Minutes Played filter (4 categories)
        self.ui.comboBox_8.clear()
        self.ui.comboBox_8.addItems([
            'All', 
            'Fresh (0-15min)', 
            'Normal (15-30min)', 
            'Heavy (30-40min)', 
            'Exhausted (40+min)'
        ])
    
    def setup_labels(self):
        """Setup filter labels"""
        try:
            # Player selection labels
            self.ui.label_20.setText("Season:")
            self.ui.label_21.setText("Team:")
            self.ui.label_22.setText("Player:")
            
            # Left side filter labels
            self.ui.label_12.setText("Home/Away:")
            self.ui.label_13.setText("Quarter:")
            self.ui.label_14.setText("Season Phase:")
            self.ui.label_15.setText("Score Margin:")
            
            # Right side filter labels
            self.ui.label_19.setText("Game Flow:")
            self.ui.label_16.setText("Rest Days:")
            self.ui.label_17.setText("Minutes Played:")
            self.ui.label_18.setText("Win/Loss Streak:")
            
        except AttributeError as e:
            print(f"⚠️ Some labels not found in UI: {e}")
    
    def setup_dropdown_hover(self):
        
        
        combos = [
            self.ui.comboBox, self.ui.comboBox_2, self.ui.comboBox_3, self.ui.comboBox_4,
            self.ui.comboBox_5, self.ui.comboBox_6, self.ui.comboBox_7, self.ui.comboBox_8,
            self.ui.comboBox_9, self.ui.comboBox_10, self.ui.comboBox_11
        ]
        
        for combo in combos:
            combo.setStyleSheet("""
                QComboBox {
                    selection-background-color: #4CAF50;
                }
            """)
                        
    def apply_complete_dropdown_styling(main_window):
        """Apply complete enhanced dropdown styling to main window"""
        
        # Enhanced button styling too
        button_style = """
        QPushButton {
            background-color: #4CAF50;
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            font-size: 12px;
            padding: 10px 20px;
            min-height: 15px;
        }
        
        QPushButton:hover {
            background-color: #45a049;
            transform: translateY(-1px);
        }
        
        QPushButton:pressed {
            background-color: #3d8b40;
            transform: translateY(0px);
        }
        
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        """
        
        # Apply to buttons
        main_window.ui.pushButton.setStyleSheet(button_style)
        main_window.ui.pushButton_2.setStyleSheet(button_style)
        
        # Label styling
        label_style = """
        QLabel {
            color: white;
            font-weight: bold;
            font-size: 11px;
        }
        """
        
        # Apply to labels
        labels = [main_window.ui.label_12, main_window.ui.label_13, main_window.ui.label_14, 
                main_window.ui.label_15, main_window.ui.label_16, main_window.ui.label_17, 
                main_window.ui.label_18, main_window.ui.label_19, main_window.ui.label_20, 
                main_window.ui.label_21, main_window.ui.label_22]
        
        for label in labels:
            label.setStyleSheet(label_style)
    def setup_export_functionality(self):
        """Add export functionality"""
        try:
            # Add export shortcut
            export_shortcut = QShortcut(QKeySequence("Ctrl+E"), self)
            export_shortcut.activated.connect(self.export_current_heatmap)
            
            # Change Reset button to Export when data is available
            self.ui.pushButton_2.setText("Export HD")
            self.ui.pushButton_2.clicked.connect(self.on_export_or_reset)
            
            
            
        except Exception as e:
            print(f"⚠️ Could not setup export functionality: {e}")
    
    def get_current_filters(self):
        """Get current filter settings"""
        return {
            'home_away': self.ui.comboBox.currentText(),
            'quarter': self.ui.comboBox_2.currentText(),
            'season_phase': self.ui.comboBox_3.currentText(),
            'score_margin': self.ui.comboBox_4.currentText(),
            'game_flow': self.ui.comboBox_5.currentText(),
            'rest_days': self.ui.comboBox_6.currentText(),
            'streak': self.ui.comboBox_7.currentText(),
            'minutes_played': self.ui.comboBox_8.currentText()
        }
    
    def reset_filters(self):
        """Reset all filters"""
        try:
            filter_combos = [
                self.ui.comboBox, self.ui.comboBox_2, self.ui.comboBox_3, self.ui.comboBox_4,
                self.ui.comboBox_5, self.ui.comboBox_6, self.ui.comboBox_7, self.ui.comboBox_8
            ]
            
            for combo in filter_combos:
                combo.setCurrentIndex(0)
            
            # Clear frame
            try:
                for child in self.ui.frame.findChildren(QLabel):
                    child.deleteLater()
                
                placeholder_label = QLabel("Optimized Resolution Court will appear here after calculating shots", self.ui.frame)
                placeholder_label.setGeometry(0, 0, self.ui.frame.width(), self.ui.frame.height())
                placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                placeholder_label.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
                placeholder_label.show()
            except Exception as e:
                print(f"⚠️ Error clearing frame: {e}")
            
            self.show_status("Filters reset")
            print("🔄 Filters reset")
            
        except Exception as e:
            print(f"❌ Error resetting filters: {e}")
    
    def setup_debug_shortcuts(self):
        """Setup debug keyboard shortcuts"""
        try:
            debug_data_shortcut = QShortcut(QKeySequence("Ctrl+Shift+D"), self)
            debug_data_shortcut.activated.connect(self.debug_shot_data_structure)
            
            test_zones_shortcut = QShortcut(QKeySequence("Ctrl+Shift+T"), self)
            test_zones_shortcut.activated.connect(self.test_zone_calculation)
            
            force_test_shortcut = QShortcut(QKeySequence("Ctrl+Shift+F"), self)
            force_test_shortcut.activated.connect(self.force_test_zones)
            
            debug_state_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
            debug_state_shortcut.activated.connect(self.debug_current_state)
            
            # Test frame display
            test_frame_shortcut = QShortcut(QKeySequence("Ctrl+Shift+T"), self)
            test_frame_shortcut.activated.connect(self.test_frame_display)
            
        
        except Exception as e:
            print(f"⚠️ Could not setup debug shortcuts: {e}")
    
    def connect_signals(self):
        """Connect UI signals"""
        self.ui.comboBox_9.currentIndexChanged.connect(self.on_season_changed_index)
        self.ui.comboBox_10.currentIndexChanged.connect(self.on_team_changed_index)
        self.ui.comboBox_11.currentIndexChanged.connect(self.on_player_changed_index)
        self.ui.pushButton.clicked.connect(self.calculate_shots)
    
    def on_season_changed_index(self, index):
        """Handle season change by index"""
        if index <= 0:
            return
        season = self.ui.comboBox_9.currentText()
        if season and season != "Select Season":
            self.on_season_changed(season)
    
    def on_team_changed_index(self, index):
        """Handle team change by index"""
        if index <= 0:
            return
        team = self.ui.comboBox_10.currentText()
        if team and team != "Select Team":
            self.on_team_changed(team)
    
    def on_player_changed_index(self, index):
        """Handle player change by index"""
        if index <= 0:
            return
        player = self.ui.comboBox_11.currentText()
        if player and player not in ["Select Player", "No players found"]:
            self.on_player_changed(player)
    
    def on_season_changed(self, season):
        """Handle season change"""
        print(f"📅 Season selected: '{season}'")
        self.current_season = season
        
        self.ui.comboBox_10.clear()
        self.ui.comboBox_10.addItems(["Select Team"])
        self.ui.comboBox_10.setEnabled(False)
        
        self.ui.comboBox_11.clear()
        self.ui.comboBox_11.addItems(["Select Player"])
        self.ui.comboBox_11.setEnabled(False)
        
        self.ui.pushButton.setEnabled(False)
        
        self.show_status("Loading teams...")
        try:
            teams = self.data_manager.get_teams_for_season_with_full_names(season)
            
            if teams:
                self.ui.comboBox_10.clear()
                self.ui.comboBox_10.addItems(["Select Team"] + teams)
                self.ui.comboBox_10.setEnabled(True)
                self.show_status(f"✅ Loaded {len(teams)} teams")
                print(f"✅ Teams loaded: {teams}")
            else:
                self.show_status("❌ No teams found for this season")
                print("❌ No teams found")
        except Exception as e:
            print(f"❌ Error loading teams: {e}")
            self.show_status(f"Error loading teams: {str(e)}")
    
    def on_team_changed(self, team):
        """Handle team change"""
        if not team or team in ["Select Team", ""]:
            return
        self.current_team = team  # Store full name
        self.current_team_abbr = self.data_manager.get_abbreviation_from_full_name(team)

        print(f"🏀 Team selected: '{team}'")
        self.current_team = team
        
        self.ui.comboBox_11.clear()
        self.ui.comboBox_11.addItems(["Loading..."])
        self.ui.comboBox_11.setEnabled(False)
        self.ui.pushButton.setEnabled(False)
        
        self.show_status(f"Loading players for {team}...")
        try:
            players = self.data_manager.get_players_for_team_season(self.current_season, team)
            
            if players:
                self.ui.comboBox_11.clear()
                self.ui.comboBox_11.addItems(["Select Player"] + players)
                self.ui.comboBox_11.setEnabled(True)
                self.show_status(f"✅ Loaded {len(players)} players for {team}")
                print(f"✅ Players loaded: {len(players)} for {team}")
            else:
                self.ui.comboBox_11.clear()
                self.ui.comboBox_11.addItems(["No players found"])
                self.show_status(f"❌ No players found for {team}")
                print(f"❌ No players found for {team}")
        except Exception as e:
            print(f"❌ Error loading players: {e}")
            self.ui.comboBox_11.clear()
            self.ui.comboBox_11.addItems(["Error loading players"])
            self.show_status(f"Error loading players: {str(e)}")
    
    def on_player_changed(self, player):
        """Handle player change"""
        if not player or player in ["Select Player", "No players found", "Loading...", "Error loading players", ""]:
            return
        
        print(f"👤 Player selected: '{player}'")
        self.current_player = player
        
        self.show_status(f"Loading shots for {player}...")
        try:
            self.shot_data = self.data_manager.load_player_shots(
                self.current_season, player
            )
            
            if self.shot_data is not None and not self.shot_data.empty:
                self.filter_engine = NBAFilterEngine(self.shot_data)
                self.ui.pushButton.setEnabled(True)
                shot_count = len(self.shot_data)
                self.show_status(f"✅ Loaded {shot_count} shots for {player}")
                print(f"✅ Loaded {shot_count} shots for {player}")
                print(f"🔘 Calculate button ENABLED")
            else:
                self.ui.pushButton.setEnabled(False)
                self.show_status(f"❌ No shots found for {player}")
                print(f"❌ No shots found for {player}")
        except Exception as e:
            print(f"❌ Error loading player shots: {e}")
            self.ui.pushButton.setEnabled(False)
            self.show_status(f"Error loading shots: {str(e)}")
    
    def calculate_shots(self):
        """Calculate shooting zones with filters and update OPTIMIZED RESOLUTION heatmap"""
        print(f"\n🎯 CALCULATING OPTIMIZED RESOLUTION SHOTS FOR {self.current_player}")
        
        try:
            if self.shot_data is None or self.shot_data.empty:
                QMessageBox.warning(self, "No Data", "No shot data available")
                return
            
            if self.filter_engine is None:
                QMessageBox.warning(self, "Error", "Filter engine not initialized")
                return
            
            filters = self.get_current_filters()
            print(f"📋 Applying filters: {filters}")
            
            filtered_data = self.filter_engine.apply_all_filters(
                self.current_player, self.current_team, filters
            )
            
            if filtered_data.empty:
                QMessageBox.information(self, "No Data", 
                                      "No shots match the selected filters.\n"
                                      "Try adjusting your filter settings.")
                return
            
            zones = self.zone_calculator.calculate_zones(filtered_data)
            summary = self.zone_calculator.get_zone_summary()
            
            # Update court visualization with OPTIMIZED RESOLUTION heatmap
            self.update_optimized_resolution_court_visualization(filtered_data, zones)
            
            total = summary['total_attempted']
            made = summary['total_made']
            pct = summary['overall_percentage']
            active_filters = sum(1 for v in filters.values() if v != 'All')
            
            self.show_status(
                f"✅ {made}/{total} shots ({pct:.1f}%) | {active_filters} filters active | Optimized Resolution Heatmap Generated!"
            )
            
            print(f"🎉 Optimized resolution analysis complete: {made}/{total} shots ({pct:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error in calculate_shots: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Calculation error: {str(e)}")
    def create_export_using_display_method(self, file_path, shot_data, zones, quality):
        """Create export using the EXACT same method as display"""
        try:
            # Create the same generator as display but for export
            frame_size = self.ui.frame.size()
            
            # Create high-res version
            export_generator = Optimal700x550HeatmapGenerator(
                shot_data, self.current_player, zones, frame_size
            )
            
            # Generate and save directly
            export_generator.create_optimal_heatmap(file_path)
            
            # Success message
            file_size = os.path.getsize(file_path) / (1024*1024)
            QMessageBox.information(self, "Export Complete", 
                                f"Heatmap exported!\n\n"
                                f"File: {os.path.basename(file_path)}\n"
                                f"Size: {file_size:.1f} MB")
            
        except Exception as e:
            print(f"❌ Export error: {e}")
            QMessageBox.critical(self, "Export Error", f"Export failed: {str(e)}")
        
    def draw_export_court(self, ax):
        """Draw professional NBA court for export - SYNCHRONIZED with display version"""
        # Professional colors
        line_color = '#2c3e50'  # Dark professional blue
        line_width = 2.5
        
        # Court outline
        court = patches.Rectangle((-250, -47.5), 500, 470, 
                                linewidth=line_width, edgecolor=line_color, 
                                facecolor='none', alpha=0.8)
        ax.add_patch(court)
        
        # Center circle
        center_circle = patches.Circle((0, 0), 60, linewidth=line_width, 
                                    edgecolor=line_color, facecolor='none', alpha=0.8)
        ax.add_patch(center_circle)
        
        # 3-point arc
        three_point_arc = patches.Arc((0, 0), 2*237.5, 2*237.5, 
                                    theta1=22, theta2=158, linewidth=line_width, 
                                    edgecolor=line_color, alpha=0.8)
        ax.add_patch(three_point_arc)
        
        # 3-point corners
        ax.plot([-220, -220], [-47.5, 92.5], color=line_color, linewidth=line_width, alpha=0.8)
        ax.plot([220, 220], [-47.5, 92.5], color=line_color, linewidth=line_width, alpha=0.8)
        
        # Paint area
        paint = patches.Rectangle((-80, -47.5), 160, 190, 
                                linewidth=line_width, edgecolor=line_color, 
                                facecolor='none', alpha=0.8)
        ax.add_patch(paint)
        
        # Free throw SEMICIRCLE (matches display version)
        ft_semicircle = patches.Arc((0, 142.5), 120, 120,  # diameter = 120 (radius 60)
                                theta1=0, theta2=180,    # 0° to 180° = top semicircle
                                linewidth=line_width, 
                                edgecolor=line_color, facecolor='none', alpha=0.8)
        ax.add_patch(ft_semicircle)
        
        # Professional basketball rim
        rim = patches.Circle((0, 0), 9, linewidth=line_width, 
                        edgecolor='#e74c3c', facecolor='#e74c3c', alpha=0.9)
        ax.add_patch(rim)
        
        # Backboard
        ax.plot([-30, 30], [-7.5, -7.5], color=line_color, linewidth=line_width * 1.2, alpha=0.8)

    def add_export_heatmap(self, ax, shot_data):
        """Add professional heatmap for export"""
        x = shot_data['x'].values
        y = shot_data['y'].values
        made = shot_data['shot_made_flag'].values
        
        # Professional color scheme
        colors = [
            '#f8f9fa', '#e3f2fd', '#bbdefb', '#90caf9', '#64b5f6',
            '#42a5f5', '#2196f3', '#1e88e5', '#1976d2', '#1565c0',
            '#0d47a1', '#ff8a65', '#ff7043', '#ff5722', '#f4511e'
        ]
        cmap_professional = LinearSegmentedColormap.from_list('professional', colors)
        
        made_x = x[made == 1]
        made_y = y[made == 1]
        
        if len(made_x) > 5:  # Only show heatmap if enough data
            # Professional bin count
            shot_count = len(made_x)
            bins = max(30, min(80, shot_count // 8))
            
            # Create smooth heatmap
            H_made, xedges, yedges = np.histogram2d(made_x, made_y, 
                                                bins=bins,
                                                range=[[-250, 250], [-47.5, 422.5]])
            
            # Professional smoothing
            sigma = max(0.8, min(1.5, shot_count / 400))
            H_made = ndimage.gaussian_filter(H_made, sigma=sigma)
            
            # Display professional heatmap
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            ax.imshow(H_made.T, extent=extent, origin='lower', 
                    cmap=cmap_professional, alpha=0.6, aspect='auto',
                    interpolation='bilinear')
        
        # Professional shot dots
        dot_size = 8
        edge_width = 0.5
        
        # Made shots - professional green
        ax.scatter(made_x, made_y, c='#27ae60', s=dot_size, alpha=0.8,
                edgecolors='white', linewidth=edge_width, label='Made')
        
        # Missed shots - professional red
        missed_x = x[made == 0]
        missed_y = y[made == 0]
        ax.scatter(missed_x, missed_y, c='#e74c3c', s=dot_size * 0.8, alpha=0.7,
                edgecolors='white', linewidth=edge_width, label='Missed')
        
        # Professional legend
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), 
                frameon=True, fancybox=True, shadow=True,
                facecolor='white', edgecolor='gray', fontsize=10)

    def add_export_zone_labels(self, ax, zones):
        """Add clean, professional zone labels for export"""
        zone_positions = {
            'Restricted Area': (0, 70),
            'In The Paint (Non-RA)': (0, 140),
            'Mid-Range': (0, 210),
            'Above the Break 3': (0, 305),
            'Left Corner 3': (-180, 50),
            'Right Corner 3': (180, 50),
        }
        
        for zone_name, stats in zones.items():
            if zone_name in zone_positions:
                x, y = zone_positions[zone_name]
                attempts = stats.get('attempted', 0)
                made = stats.get('made', 0)
                pct = stats.get('percentage', 0)
                
                if attempts >= 3:
                    label_text = f"{pct:.1f}%\n{made}/{attempts}"
                    
                    # Professional color coding
                    if pct >= 50:
                        bg_color = '#27ae60'  # Professional green
                        text_color = 'white'
                    elif pct >= 40:
                        bg_color = '#f39c12'  # Professional orange
                        text_color = 'white'
                    elif pct >= 30:
                        bg_color = '#e67e22'  # Professional dark orange
                        text_color = 'white'
                    else:
                        bg_color = '#e74c3c'  # Professional red
                        text_color = 'white'
                    
                    # Clean professional styling
                    ax.text(x, y, label_text, fontsize=11, fontweight='bold',
                        ha='center', va='center', color=text_color,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor=bg_color,
                                edgecolor='white', linewidth=1.5, alpha=0.95))
                
    def update_optimized_resolution_court_visualization(self, shot_data: pd.DataFrame, zones: Dict[str, Dict[str, Union[int, float]]]):
        """Update court with OPTIMAL 700x550 heatmap"""
        try:
            print(f"🎨 Creating OPTIMAL heatmap for 700x550 frame ({len(shot_data)} shots)")
            
            self.show_loading_message()
            
            frame_size = self.ui.frame.size()
            print(f"📐 Frame size: {frame_size.width()}x{frame_size.height()}")
            
            # Use optimal generator
            self.viz_thread = Optimal700x550HeatmapGenerator(
                shot_data, self.current_player, zones, frame_size
            )
            self.viz_thread.image_ready.connect(self.update_court_image_optimal)
            self.viz_thread.start()
            
            print("✅ Optimal generation started")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
    def replace_court_image_with_ultra_quality_heatmap(self, shot_data, zones):
        """Generate heatmap with ULTRA QUALITY resolution for maximum visual fidelity"""
        print("🎨 Generating ULTRA QUALITY RESOLUTION heatmap...")
        
        self.show_loading_message()
        
        # Get actual frame size
        frame_size = self.ui.frame.size()
        print(f"📐 Frame size: {frame_size.width()}x{frame_size.height()}")
        
        # Create ultra high quality generator
        self.viz_thread = UltraHighQualityHeatmapGenerator(
            shot_data, self.current_player, zones, frame_size
        )
        self.viz_thread.image_ready.connect(self.update_court_image_ultra_quality)
        self.viz_thread.start()
    
    def show_loading_message(self):
        """Show loading message in the frame"""
        try:
            # Clear any existing content
            for child in self.ui.frame.findChildren(QLabel):
                child.deleteLater()
            
            # Create loading label
            loading_label = QLabel("🎨 Generating Heatmap...\nPlease wait (~2-3 seconds)...", self.ui.frame)
            loading_label.setGeometry(0, 0, self.ui.frame.width(), self.ui.frame.height())
            loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            loading_label.setStyleSheet("""
                color: white; 
                font-size: 16px; 
                font-weight: bold; 
                background-color: rgba(0, 0, 0, 0.7);
                border-radius: 10px;
                padding: 20px;
            """)
            loading_label.show()
            
            # Force UI update
            QApplication.processEvents()
            
        except Exception as e:
            print(f"⚠️ Error showing loading message: {e}")
    
    def update_court_image_optimal(self, image_path):
        """Display optimal quality heatmap with perfect scaling"""
        try:
            print(f"🖼️ [OPTIMAL] Loading: {image_path}")
            
            if not os.path.exists(image_path):
                self.show_error_message("Heatmap file not found")
                return
            
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                self.show_error_message("Failed to load heatmap")
                return
            
            print(f"✅ Loaded: {pixmap.width()}x{pixmap.height()}")
            print(f"📏 Frame: {self.ui.frame.width()}x{self.ui.frame.height()}")
            
            # Clear frame
            for child in self.ui.frame.findChildren(QLabel):
                child.deleteLater()
            QApplication.processEvents()
            
            # Create label
            heatmap_label = QLabel(self.ui.frame)
            heatmap_label.setGeometry(0, 0, self.ui.frame.width(), self.ui.frame.height())
            
            # Scale with optimal quality
            scaled_pixmap = pixmap.scaled(
                self.ui.frame.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            print(f"📏 Scaled to: {scaled_pixmap.width()}x{scaled_pixmap.height()}")
            
            heatmap_label.setPixmap(scaled_pixmap)
            heatmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            heatmap_label.show()
            
            self.ui.frame.update()
            QApplication.processEvents()
            
            print("✅ [OPTIMAL] Perfect 700x550 heatmap displayed!")
            
            self.current_heatmap_label = heatmap_label
            
            # Clean up
            try:
                import time
                time.sleep(0.1)
                if os.path.exists(image_path):
                    os.unlink(image_path)
            except:
                pass
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    
    def show_error_message(self, error_text):
        """Show error message in the frame"""
        try:
            # Clear frame
            for child in self.ui.frame.findChildren(QLabel):
                child.deleteLater()
            
            # Create error label
            error_label = QLabel(f"❌ {error_text}\n\nTry clicking Calculate again", self.ui.frame)
            error_label.setGeometry(0, 0, self.ui.frame.width(), self.ui.frame.height())
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_label.setStyleSheet("""
                color: #ff6b6b; 
                font-size: 14px; 
                font-weight: bold; 
                background-color: rgba(0, 0, 0, 0.8);
                border-radius: 10px;
                padding: 20px;
                border: 2px solid #ff6b6b;
            """)
            error_label.show()
            
        except Exception as e:
            print(f"⚠️ Error showing error message: {e}")
    
    def test_frame_display(self):
        """Test method to verify frame can display content"""
        try:
            print("🧪 Testing frame display capability...")
            
            # Clear frame
            for child in self.ui.frame.findChildren(QLabel):
                child.deleteLater()
            
            # Create test label
            test_label = QLabel("✅ FRAME TEST\nIf you see this, the frame works!", self.ui.frame)
            test_label.setGeometry(0, 0, self.ui.frame.width(), self.ui.frame.height())
            test_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            test_label.setStyleSheet("""
                color: #4CAF50; 
                font-size: 18px; 
                font-weight: bold; 
                background-color: rgba(0, 0, 0, 0.8);
                border-radius: 10px;
                padding: 20px;
                border: 2px solid #4CAF50;
            """)
            test_label.show()
            
            print("✅ Test frame display complete")
            
        except Exception as e:
            print(f"❌ Frame test error: {e}")
    
    def on_export_or_reset(self):
        """Handle export or reset based on current state"""
        if self.shot_data is not None and not self.shot_data.empty:
            # If we have data, this becomes export
            self.export_current_heatmap()
        else:
            # If no data, this resets filters
            self.reset_filters()
    
    def export_current_heatmap(self):
        """Export the current heatmap in high resolution"""
        try:
            if self.shot_data is None or self.shot_data.empty:
                QMessageBox.warning(self, "No Data", "Load a player first to export heatmap")
                return
            
            if self.filter_engine is None:
                QMessageBox.warning(self, "No Data", "No analysis data available for export")
                return
            
            print("🖼️ Preparing high resolution export...")
            
            # Get current filters
            filters = self.get_current_filters()
            
            # Apply filters to data
            filtered_data = self.filter_engine.apply_all_filters(
                self.current_player, self.current_team, filters
            )
            
            if filtered_data.empty:
                QMessageBox.warning(self, "No Data", "No shots match current filters")
                return
            
            # Calculate zones
            zones = self.zone_calculator.calculate_zones(filtered_data)
            
            # Show export dialog
            self.show_export_dialog(filtered_data, zones)
            
        except Exception as e:
            print(f"❌ Export preparation error: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to prepare export: {str(e)}")
    
    def show_export_dialog(self, shot_data, zones):
        """Show export dialog with options"""
        try:
            # Create export dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Export High-Resolution Heatmap")
            dialog.setFixedSize(400, 200)
            
            layout = QVBoxLayout()
            
            # Title
            title = QLabel("Export High-Resolution Heatmap")
            title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
            layout.addWidget(title)
            
            # Quality selection
            quality_layout = QHBoxLayout()
            quality_layout.addWidget(QLabel("Quality:"))
            
            quality_combo = QComboBox()
            quality_combo.addItems([
                "Ultra High (450 DPI) - Best Quality",
                "High (300 DPI) - Good Quality", 
                "Standard (220 DPI) - Fast Export"
            ])
            quality_layout.addWidget(quality_combo)
            layout.addLayout(quality_layout)
            
            # Info
            info = QLabel(f"Player: {self.current_player}\nShots: {len(shot_data)}\nFilters: {sum(1 for v in self.get_current_filters().values() if v != 'All')} active")
            info.setStyleSheet("margin: 10px 0px; padding: 10px; background-color: rgba(0,0,0,0.1); border-radius: 5px;")
            layout.addWidget(info)
            
            # Buttons
            button_layout = QHBoxLayout()
            
            export_btn = QPushButton("Export Heatmap")
            export_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px; font-weight: bold;")
            export_btn.clicked.connect(lambda: self.do_export(dialog, quality_combo.currentIndex(), shot_data, zones))
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(dialog.reject)
            
            button_layout.addWidget(cancel_btn)
            button_layout.addWidget(export_btn)
            layout.addLayout(button_layout)
            
            dialog.setLayout(layout)
            dialog.exec()
            
        except Exception as e:
            print(f"❌ Export dialog error: {e}")
    
    def do_export(self, dialog, quality_index, shot_data, zones):
        """Perform the actual export"""
        try:
            # Quality settings
            quality_settings = [
                {'dpi': 450, 'size': (24, 18), 'name': 'Ultra'},
                {'dpi': 300, 'size': (16, 12), 'name': 'High'},
                {'dpi': 220, 'size': (12, 9), 'name': 'Standard'}
            ]
            
            quality = quality_settings[quality_index]
            
            # Get save location
            default_name = f"{self.current_player}_{self.current_season}_heatmap_{quality['name']}.png"
            file_path, _ = QFileDialog.getSaveFileName(
                dialog, "Save High-Resolution Heatmap", 
                default_name,
                "PNG files (*.png);;All files (*.*)"
            )
            
            if file_path:
                dialog.accept()
                
                # Show progress
                self.show_status(f"🖼️ Exporting {quality['name']} quality heatmap... Please wait...")
                
                # Use the correct method name - create_professional_export instead of create_custom_export
                self.create_export_using_display_method(file_path, shot_data, zones, quality)
                
        except Exception as e:
            print(f"❌ Export execution error: {e}")
            QMessageBox.critical(dialog, "Export Error", f"Export failed: {str(e)}")
    
    def create_professional_export(self, file_path, shot_data, zones, quality):
        """Create high-resolution export that matches display exactly"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            
            print(f"🖼️ Creating HIGH-RES {quality['name']} export...")
            
            # High resolution figure
            fig_width, fig_height = 16, 12  # Large size for high quality
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=quality['dpi'])
            
            # SAME styling as display
            fig.patch.set_facecolor('#1a1a1a')  # Dark background
            ax.set_facecolor('#0a1929')         # Dark court
            
            plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08)
            
            # Draw court with display colors
            line_width = max(3, quality['dpi'] / 100)
            line_color = '#ffffff'  # White lines like display
            
            # Court elements (same as display)
            court = patches.Rectangle((-250, -47.5), 500, 470, 
                                    linewidth=line_width, edgecolor=line_color, 
                                    facecolor='none', alpha=0.9)
            ax.add_patch(court)
            
            center_circle = patches.Circle((0, 0), 60, linewidth=line_width, 
                                        edgecolor=line_color, facecolor='none', alpha=0.9)
            ax.add_patch(center_circle)
            
            three_point_arc = patches.Arc((0, 0), 2*237.5, 2*237.5, 
                                        theta1=22, theta2=158, linewidth=line_width, 
                                        edgecolor=line_color, alpha=0.9)
            ax.add_patch(three_point_arc)
            
            ax.plot([-220, -220], [-47.5, 92.5], color=line_color, linewidth=line_width, alpha=0.9)
            ax.plot([220, 220], [-47.5, 92.5], color=line_color, linewidth=line_width, alpha=0.9)
            
            paint = patches.Rectangle((-80, -47.5), 160, 190, 
                                    linewidth=line_width, edgecolor=line_color, 
                                    facecolor='none', alpha=0.9)
            ax.add_patch(paint)
            
            # Free throw semicircle (like display)
            ft_semicircle = patches.Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                                    linewidth=line_width, edgecolor=line_color, 
                                    facecolor='none', alpha=0.9)
            ax.add_patch(ft_semicircle)
            
            # Rim
            rim = patches.Circle((0, 0), max(8, quality['dpi']/40), linewidth=line_width, 
                            edgecolor='#ff6b35', facecolor='#ff6b35', alpha=1.0)
            ax.add_patch(rim)
            
            ax.plot([-30, 30], [-7.5, -7.5], color=line_color, linewidth=line_width*1.5, alpha=0.9)
            
            # Add heatmap and shots with display colors
            if not shot_data.empty:
                x, y = shot_data['x'].values, shot_data['y'].values
                made = shot_data['shot_made_flag'].values
                
                # Display-style heatmap
                made_x, made_y = x[made == 1], y[made == 1]
                if len(made_x) > 0:
                    colors_hot = ['#001122', '#003366', '#0066aa', '#0099ee', '#33ccff', 
                                '#66ffcc', '#ccff66', '#ffcc33', '#ff9900', '#ff4400', '#cc0000']
                    cmap_hot = LinearSegmentedColormap.from_list('export_hot', colors_hot)
                    
                    bins = max(50, min(100, quality['dpi'] // 5))
                    H_made, xedges, yedges = np.histogram2d(made_x, made_y, bins=bins,
                                                        range=[[-250, 250], [-47.5, 422.5]])
                    H_made = ndimage.gaussian_filter(H_made, sigma=quality['dpi']/200)
                    
                    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                    ax.imshow(H_made.T, extent=extent, origin='lower', 
                            cmap=cmap_hot, alpha=0.75, aspect='auto', interpolation='bilinear')
                
                # Display-style shot dots
                dot_size = max(10, quality['dpi'] / 30)
                edge_width = max(1, quality['dpi'] / 300)
                
                ax.scatter(made_x, made_y, c='#00FF7F', s=dot_size, alpha=0.9,
                        edgecolors='white', linewidth=edge_width)
                ax.scatter(x[made == 0], y[made == 0], c='#FF4444', s=dot_size*0.85, alpha=0.8,
                        edgecolors='white', linewidth=edge_width)
            
            # Add zones with display colors
            if zones:
                zone_positions = {
                    'Restricted Area': (0, 70), 'In The Paint (Non-RA)': (0, 140),
                    'Mid-Range': (0, 210), 'Above the Break 3': (0, 305),
                    'Left Corner 3': (-180, 50), 'Right Corner 3': (180, 50)
                }
                
                font_size = max(12, quality['dpi'] / 25)
                for zone_name, stats in zones.items():
                    if zone_name in zone_positions and stats.get('attempted', 0) > 0:
                        x, y = zone_positions[zone_name]
                        pct = stats.get('percentage', 0)
                        
                        if pct >= 50: bg_color = '#00C853'
                        elif pct >= 40: bg_color = '#FFB300'  
                        elif pct >= 30: bg_color = '#FF8F00'
                        else: bg_color = '#D32F2F'
                        
                        ax.text(x, y, f"{pct:.1f}%\n{stats['made']}/{stats['attempted']}", 
                            fontsize=font_size, fontweight='bold', ha='center', va='center', 
                            color='white', bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color,
                                                    edgecolor='white', linewidth=2, alpha=0.95))
            
            # Court dimensions and styling
            ax.set_xlim(-260, 260)
            ax.set_ylim(-50, 430)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Title
            plt.title(f'{self.current_player} - {self.current_season} Shot Analysis', 
                    color='white', fontsize=max(18, quality['dpi']/20), fontweight='bold', pad=20)
            
            # Save
            plt.savefig(file_path, dpi=quality['dpi'], bbox_inches='tight', 
                    facecolor='#1a1a1a', edgecolor='none', pad_inches=0.1)
            plt.close()
            
            # Success message
            file_size = os.path.getsize(file_path) / (1024*1024)
            QMessageBox.information(self, "Export Complete", 
                                f"High-res heatmap exported!\n\n"
                                f"Size: {file_size:.1f} MB\n"
                                f"Quality: {quality['name']} ({quality['dpi']} DPI)")
            
        except Exception as e:
            print(f"❌ Export error: {e}")
            QMessageBox.critical(self, "Export Error", f"Export failed: {str(e)}")
    
    def debug_shot_data_structure(self):
        """Debug the structure of loaded shot data"""
        if self.shot_data is None or self.shot_data.empty:
            print("❌ No shot data loaded")
            return
        
        print(f"\n🔍 DEBUG: Shot Data Structure")
        print("=" * 50)
        print(f"Data shape: {self.shot_data.shape}")
        print(f"Columns: {list(self.shot_data.columns)}")
        
        zone_columns = [col for col in self.shot_data.columns if 'zone' in col.lower()]
        print(f"\nZone columns found: {zone_columns}")
        
        coord_columns = [col for col in self.shot_data.columns if col.lower() in ['x', 'y', 'loc_x', 'loc_y']]
        print(f"Coordinate columns found: {coord_columns}")
        
        for col in zone_columns[:3]:
            print(f"\n📊 {col} sample values:")
            values = self.shot_data[col].value_counts().head(5)
            for value, count in values.items():
                print(f"   '{value}': {count} shots")
        
        if 'shot_made_flag' in self.shot_data.columns:
            made_stats = self.shot_data['shot_made_flag'].value_counts()
            print(f"\nShot results: {dict(made_stats)}")
            total_pct = (made_stats.get(1, 0) / len(self.shot_data) * 100)
            print(f"Overall shooting: {total_pct:.1f}%")
        
        print("=" * 50)
    
    def test_zone_calculation(self):
        """Test zone calculation with current data"""
        if self.shot_data is None or self.shot_data.empty:
            print("❌ No shot data to test")
            return
        
        print(f"\n🧪 TESTING: Zone Calculation")
        print("=" * 40)
        
        test_calculator = ShotZoneCalculator()
        zones = test_calculator.calculate_zones(self.shot_data)
        
        print(f"Zones calculated: {len(zones)}")
        for zone_name, stats in zones.items():
            print(f"   {zone_name}: {stats}")
        
        print("=" * 40)
    
    def force_test_zones(self):
        """Force test zone display with dummy data"""
        print(f"\n🧪 FORCE TESTING: Optimized Resolution Zone Display")
        
        dummy_zones = {
            'Above the Break 3': {'attempted': 315, 'made': 125, 'percentage': 39.7},
            'Left Corner 3': {'attempted': 42, 'made': 22, 'percentage': 52.4},
            'Right Corner 3': {'attempted': 36, 'made': 20, 'percentage': 55.6},
            'Restricted Area': {'attempted': 180, 'made': 115, 'percentage': 63.9},
            'In The Paint (Non-RA)': {'attempted': 217, 'made': 109, 'percentage': 50.2},
            'Mid-Range': {'attempted': 183, 'made': 76, 'percentage': 41.5}
        }
        
        print("Using dummy NBA zone data for optimized resolution testing:")
        for zone, stats in dummy_zones.items():
            print(f"   {zone}: {stats}")
        
        # Create dummy shot data for optimized resolution heatmap
        dummy_data = pd.DataFrame({
            'x': np.random.uniform(-250, 250, 1000),
            'y': np.random.uniform(-47.5, 422.5, 1000),
            'shot_made_flag': np.random.choice([0, 1], 1000),
            'shot_zone_basic': np.random.choice(list(dummy_zones.keys()), 1000)
        })
        
        self.update_optimized_resolution_court_visualization(dummy_data, dummy_zones)
        print("✅ Dummy zones applied to optimized resolution heatmap")
    
    def debug_current_state(self):
        """Debug current application state"""
        print("\n🔍 DEBUG: Current Application State")
        print("=" * 50)
        print(f"Current Season: {self.current_season}")
        print(f"Current Team: {self.current_team}")
        print(f"Current Player: {self.current_player}")
        print(f"Shot Data: {self.shot_data.shape if self.shot_data is not None else 'None'}")
        print(f"Filter Engine: {'Initialized' if self.filter_engine else 'None'}")
        print(f"Calculate Button Enabled: {self.ui.pushButton.isEnabled()}")
        
        try:
            season_text = self.ui.comboBox_9.currentText()
            team_text = self.ui.comboBox_10.currentText()
            player_text = self.ui.comboBox_11.currentText()
            print(f"UI Season: {season_text}")
            print(f"UI Team: {team_text}")
            print(f"UI Player: {player_text}")
        except Exception as e:
            print(f"Error getting UI selections: {e}")
        
        print("=" * 50)
    
    def show_status(self, message: str):
        """Show status message"""
        print(f"📢 STATUS: {message}")
        try:
            self.ui.statusbar.showMessage(message, 5000)
        except AttributeError:
            print("⚠️ Status bar not found")
    
    def closeEvent(self, event):
        """Handle application close event"""
        try:
            print("👋 Application closing...")
            # Clean up any background threads
            if hasattr(self, 'viz_thread') and self.viz_thread.isRunning():
                self.viz_thread.quit()
                self.viz_thread.wait()
            event.accept()
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
            event.accept()

def main():
    """Main entry point"""
    print("🚀 Starting NBA Shot Analyzer with OPTIMIZED RESOLUTION System...")
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    try:
        window = NBAShotAnalyzer()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        QMessageBox.critical(None, "Error", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()