"""
Simple Pygame UI for the Games Kernel
"""

import pygame
import sys
import os
from typing import List, Dict, Any, Optional, Tuple

# Import the games kernel
from games_kernel import GameEngine, Player
from mahjong_game import MahjongGame, MahjongPlayer, Wind, TileType, Tile
from chess_game import ChessGame, ChessPlayer, PieceColor, PieceType, ChessPiece
from go_game import GoGame, GoPlayer, Stone

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
BOARD_SIZE = 600
MARGIN = 50

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_BROWN = (222, 184, 135)
DARK_BROWN = (160, 120, 80)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class GameGUI:
    """Base class for game GUIs"""
    
    def __init__(self, game):
        self.game = game
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(f"Games Kernel - {game.name}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        self.title_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.selected = None
    
    def draw_text(self, text, font, color, x, y, align="left"):
        """Draw text on the screen"""
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        
        if align == "center":
            text_rect.center = (x, y)
        elif align == "right":
            text_rect.right, text_rect.top = x, y
        else:
            text_rect.left, text_rect.top = x, y
        
        self.screen.blit(text_surface, text_rect)
        return text_rect
    
    def draw_game_info(self):
        """Draw general game information"""
        # Draw game title
        self.draw_text(self.game.name, self.title_font, BLACK, SCREEN_WIDTH // 2, 20, align="center")
        
        # Draw current player
        player_text = f"Current Player: {self.game.current_player.name}"
        self.draw_text(player_text, self.font, BLACK, MARGIN, MARGIN // 2)
        
        # Draw game status
        if self.game.is_game_over:
            if self.game.winner:
                status_text = f"Game Over - Winner: {self.game.winner.name}"
            else:
                status_text = "Game Over - Draw"
            self.draw_text(status_text, self.font, RED, SCREEN_WIDTH - MARGIN, MARGIN // 2, align="right")
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            self.handle_game_events(event)
    
    def handle_game_events(self, event):
        """Handle game-specific events"""
        pass
    
    def draw(self):
        """Draw the game"""
        self.screen.fill(WHITE)
        self.draw_game_info()
        self.draw_game_content()
        pygame.display.flip()
    
    def draw_game_content(self):
        """Draw game-specific content"""
        pass
    
    def run(self):
        """Main game loop"""
        while True:
            self.handle_events()
            self.draw()
            self.clock.tick(30)

class MahjongGUI(GameGUI):
    """GUI for Mahjong game"""
    
    def __init__(self, game):
        super().__init__(game)
        
        # Load images for tiles (placeholder)
        self.tile_images = self._load_tile_images()
        
        # Tile dimensions
        self.tile_width = 40
        self.tile_height = 60
        
        # Selected tile index
        self.selected_tile_idx = None
    
    def _load_tile_images(self):
        """Load tile images (placeholder - would load real images in a full implementation)"""
        # Create placeholder tiles for each type
        images = {}
        
        # Suited tiles
        for suit in [TileType.DOTS, TileType.BAMBOO, TileType.CHARACTERS]:
            for value in range(1, 10):
                key = (suit, value)
                # Create a surface with the suit and value text
                img = pygame.Surface((self.tile_width, self.tile_height))
                img.fill(WHITE)
                pygame.draw.rect(img, BLACK, (0, 0, self.tile_width, self.tile_height), 2)
                
                # Add text for suit and value
                font = pygame.font.SysFont("Arial", 14)
                value_text = font.render(str(value), True, BLACK)
                suit_text = font.render(suit.value[0], True, BLACK)
                
                img.blit(value_text, (self.tile_width // 2 - value_text.get_width() // 2, 5))
                img.blit(suit_text, (self.tile_width // 2 - suit_text.get_width() // 2, 25))
                
                images[key] = img
        
        # Honor tiles (winds and dragons)
        for wind in Wind:
            key = (TileType.WIND, wind)
            img = pygame.Surface((self.tile_width, self.tile_height))
            img.fill(WHITE)
            pygame.draw.rect(img, BLACK, (0, 0, self.tile_width, self.tile_height), 2)
            
            font = pygame.font.SysFont("Arial", 14)
            text = font.render(wind.value[0], True, BLUE)
            img.blit(text, (self.tile_width // 2 - text.get_width() // 2, self.tile_height // 2 - text.get_height() // 2))
            
            images[key] = img
        
        return images
    
    def draw_game_content(self):
        """Draw the Mahjong game content"""
        # Draw the wall (simplified)
        wall_text = f"Wall: {len(self.game.wall)} tiles"
        self.draw_text(wall_text, self.font, BLACK, MARGIN, MARGIN * 2)
        
        # Draw dora indicators
        dora_text = "Dora Indicators: " + ", ".join(str(tile) for tile in self.game.dora_indicators)
        self.draw_text(dora_text, self.font, BLACK, MARGIN, MARGIN * 3)
        
        # Draw current discard
        discard_text = f"Current Discard: {self.game.current_discard}"
        self.draw_text(discard_text, self.font, BLACK, MARGIN, MARGIN * 4)
        
        # Draw player hands
        self.draw_player_hands()
        
        # Draw action buttons
        self.draw_action_buttons()
    
    def draw_player_hands(self):
        """Draw the hands of all players"""
        for i, player in enumerate(self.game.players):
            # Position for each player's hand
            positions = [
                (SCREEN_WIDTH // 2, SCREEN_HEIGHT - MARGIN * 2),  # Bottom (current player)
                (MARGIN * 2, SCREEN_HEIGHT // 2),  # Left
                (SCREEN_WIDTH // 2, MARGIN * 5),  # Top
                (SCREEN_WIDTH - MARGIN * 2, SCREEN_HEIGHT // 2)   # Right
            ]
            
            x, y = positions[i]
            
            # Highlight current player
            color = RED if i == self.game.current_player_idx else BLACK
            self.draw_text(f"{player.name} ({player.wind.value})", self.font, color, x, y - 20, align="center")
            
            # If this is the current player, show the full hand
            if i == self.game.current_player_idx:
                self.draw_player_tiles(player, x, y)
            else:
                # For other players, just show the number of tiles
                tile_count = len(player.hand.tiles)
                self.draw_text(f"{tile_count} tiles", self.font, BLACK, x, y, align="center")
    
    def draw_player_tiles(self, player, x, y):
        """Draw a player's tiles"""
        tiles = player.hand.tiles
        total_width = len(tiles) * self.tile_width
        start_x = x - total_width // 2
        
        for i, tile in enumerate(tiles):
            tile_x = start_x + i * self.tile_width
            
            # Draw a background rectangle to show selection
            if i == self.selected_tile_idx:
                pygame.draw.rect(self.screen, GREEN, (tile_x - 2, y - 2, self.tile_width + 4, self.tile_height + 4))
            
            # Draw the tile
            if (tile.type, tile.value) in self.tile_images:
                self.screen.blit(self.tile_images[(tile.type, tile.value)], (tile_x, y))
            else:
                # Fallback for missing images
                img = pygame.Surface((self.tile_width, self.tile_height))
                img.fill(WHITE)
                pygame.draw.rect(img, BLACK, (0, 0, self.tile_width, self.tile_height), 2)
                self.screen.blit(img, (tile_x, y))
                self.draw_text(str(tile), self.font, BLACK, tile_x + 5, y + 20)
    
    def draw_action_buttons(self):
        """Draw action buttons for the current player"""
        button_width = 100
        button_height = 30
        button_margin = 10
        start_x = MARGIN
        start_y = SCREEN_HEIGHT - MARGIN * 4
        
        buttons = ["Discard", "Chi", "Pon", "Kan", "Win"]
        self.button_rects = []
        
        for i, text in enumerate(buttons):
            button_x = start_x + i * (button_width + button_margin)
            button_rect = pygame.Rect(button_x, start_y, button_width, button_height)
            
            # Draw the button
            pygame.draw.rect(self.screen, GRAY, button_rect)
            pygame.draw.rect(self.screen, BLACK, button_rect, 2)
            self.draw_text(text, self.font, BLACK, button_rect.centerx, button_rect.centery, align="center")
            
            self.button_rects.append(button_rect)
    
    def handle_game_events(self, event):
        """Handle Mahjong-specific events"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if a tile was clicked
            player = self.game.current_player
            tiles = player.hand.tiles
            total_width = len(tiles) * self.tile_width
            x, y = SCREEN_WIDTH // 2, SCREEN_HEIGHT - MARGIN * 2
            start_x = x - total_width // 2
            
            for i in range(len(tiles)):
                tile_x = start_x + i * self.tile_width
                tile_rect = pygame.Rect(tile_x, y, self.tile_width, self.tile_height)
                
                if tile_rect.collidepoint(event.pos):
                    self.selected_tile_idx = i
                    break
            
            # Check if an action button was clicked
            for i, button_rect in enumerate(self.button_rects):
                if button_rect.collidepoint(event.pos):
                    self.handle_action_button(i)
    
    def handle_action_button(self, button_idx):
        """Handle action button clicks"""
        if button_idx == 0:  # Discard
            if self.selected_tile_idx is not None:
                move = {"type": "discard", "tile_idx": self.selected_tile_idx}
                if self.game.is_valid_move(move):
                    self.game.make_move(move)
                    self.selected_tile_idx = None
        
        elif button_idx == 4:  # Win
            move = {"type": "win", "from_discard": False}
            if self.game.is_valid_move(move):
                self.game.make_move(move)

class ChessGUI(GameGUI):
    """GUI for Chess game"""
    
    def __init__(self, game):
        super().__init__(game)
        
        # Chess square size
        self.square_size = BOARD_SIZE // 8
        
        # Board offset
        self.board_offset_x = (SCREEN_WIDTH - BOARD_SIZE) // 2
        self.board_offset_y = (SCREEN_HEIGHT - BOARD_SIZE) // 2
        
        # Selected square
        self.selected_square = None
        
        # Load chess piece images (placeholder)
        self.piece_images = self._load_piece_images()
    
    def _load_piece_images(self):
        """Load chess piece images (placeholder - would load real images in a full implementation)"""
        images = {}
        
        for color in [PieceColor.WHITE, PieceColor.BLACK]:
            for piece_type in PieceType:
                if piece_type == PieceType.EMPTY:
                    continue
                
                # Create a placeholder image
                img = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                
                # Draw the piece
                if piece_type == PieceType.PAWN:
                    self._draw_pawn(img, color)
                elif piece_type == PieceType.KNIGHT:
                    self._draw_knight(img, color)
                elif piece_type == PieceType.BISHOP:
                    self._draw_bishop(img, color)
                elif piece_type == PieceType.ROOK:
                    self._draw_rook(img, color)
                elif piece_type == PieceType.QUEEN:
                    self._draw_queen(img, color)
                elif piece_type == PieceType.KING:
                    self._draw_king(img, color)
                
                images[(piece_type, color)] = img
        
        return images
    
    def _draw_pawn(self, surface, color):
        """Draw a pawn (placeholder)"""
        piece_color = WHITE if color == PieceColor.WHITE else BLACK
        pygame.draw.circle(surface, piece_color, (self.square_size // 2, self.square_size // 2), self.square_size // 4)
        pygame.draw.circle(surface, BLACK, (self.square_size // 2, self.square_size // 2), self.square_size // 4, 2)
        font = pygame.font.SysFont("Arial", self.square_size // 3)
        text = font.render("P", True, BLACK if color == PieceColor.WHITE else WHITE)
        surface.blit(text, (self.square_size // 2 - text.get_width() // 2, 
                           self.square_size // 2 - text.get_height() // 2))
    
    def _draw_knight(self, surface, color):
        """Draw a knight (placeholder)"""
        piece_color = WHITE if color == PieceColor.WHITE else BLACK
        pygame.draw.circle(surface, piece_color, (self.square_size // 2, self.square_size // 2), self.square_size // 3)
        pygame.draw.circle(surface, BLACK, (self.square_size // 2, self.square_size // 2), self.square_size // 3, 2)
        font = pygame.font.SysFont("Arial", self.square_size // 3)
        text = font.render("N", True, BLACK if color == PieceColor.WHITE else WHITE)
        surface.blit(text, (self.square_size // 2 - text.get_width() // 2, 
                           self.square_size // 2 - text.get_height() // 2))
    
    def _draw_bishop(self, surface, color):
        """Draw a bishop (placeholder)"""
        piece_color = WHITE if color == PieceColor.WHITE else BLACK
        pygame.draw.circle(surface, piece_color, (self.square_size // 2, self.square_size // 2), self.square_size // 3)
        pygame.draw.circle(surface, BLACK, (self.square_size // 2, self.square_size // 2), self.square_size // 3, 2)
        font = pygame.font.SysFont("Arial", self.square_size // 3)
        text = font.render("B", True, BLACK if color == PieceColor.WHITE else WHITE)
        surface.blit(text, (self.square_size // 2 - text.get_width() // 2, 
                           self.square_size // 2 - text.get_height() // 2))
    
    def _draw_rook(self, surface, color):
        """Draw a rook (placeholder)"""
        piece_color = WHITE if color == PieceColor.WHITE else BLACK
        pygame.draw.circle(surface, piece_color, (self.square_size // 2, self.square_size // 2), self.square_size // 3)
        pygame.draw.circle(surface, BLACK, (self.square_size // 2, self.square_size // 2), self.square_size // 3, 2)
        font = pygame.font.SysFont("Arial", self.square_size // 3)
        text = font.render("R", True, BLACK if color == PieceColor.WHITE else WHITE)
        surface.blit(text, (self.square_size // 2 - text.get_width() // 2, 
                           self.square_size // 2 - text.get_height() // 2))
    
    def _draw_queen(self, surface, color):
        """Draw a queen (placeholder)"""
        piece_color = WHITE if color == PieceColor.WHITE else BLACK
        pygame.draw.circle(surface, piece_color, (self.square_size // 2, self.square_size // 2), self.square_size // 3)
        pygame.draw.circle(surface, BLACK, (self.square_size // 2, self.square_size // 2), self.square_size // 3, 2)
        font = pygame.font.SysFont("Arial", self.square_size // 3)
        text = font.render("Q", True, BLACK if color == PieceColor.WHITE else WHITE)
        surface.blit(text, (self.square_size // 2 - text.get_width() // 2, 
                           self.square_size // 2 - text.get_height() // 2))
    
    def _draw_king(self, surface, color):
        """Draw a king (placeholder)"""
        piece_color = WHITE if color == PieceColor.WHITE else BLACK
        pygame.draw.circle(surface, piece_color, (self.square_size // 2, self.square_size // 2), self.square_size // 3)
        pygame.draw.circle(surface, BLACK, (self.square_size // 2, self.square_size // 2), self.square_size // 3, 2)
        font = pygame.font.SysFont("Arial", self.square_size // 3)
        text = font.render("K", True, BLACK if color == PieceColor.WHITE else WHITE)
        surface.blit(text, (self.square_size // 2 - text.get_width() // 2, 
                           self.square_size // 2 - text.get_height() // 2))
    
    def draw_game_content(self):
        """Draw the Chess game content"""
        # Draw the board
        self.draw_board()
        
        # Draw game status
        if self.game.check_status:
            self.draw_text("CHECK!", self.font, RED, SCREEN_WIDTH - MARGIN, MARGIN * 2, align="right")
        
        # Draw captured pieces
        white_captured = [str(p) for p in self.game.players[0].captured_pieces]
        black_captured = [str(p) for p in self.game.players[1].captured_pieces]
        
        self.draw_text(f"White Captured: {' '.join(white_captured)}", self.font, BLACK, MARGIN, MARGIN * 2)
        self.draw_text(f"Black Captured: {' '.join(black_captured)}", self.font, BLACK, MARGIN, MARGIN * 3)
    
    def draw_board(self):
        """Draw the chess board"""
        # Draw the squares
        for row in range(8):
            for col in range(8):
                x = self.board_offset_x + col * self.square_size
                y = self.board_offset_y + (7 - row) * self.square_size  # Flip y-axis to match chess coordinates
                
                # Determine square color
                color = WHITE if (row + col) % 2 == 0 else GRAY
                
                # Draw the square
                pygame.draw.rect(self.screen, color, (x, y, self.square_size, self.square_size))
                
                # Highlight selected square
                if self.selected_square == (col, row):
                    pygame.draw.rect(self.screen, GREEN, (x, y, self.square_size, self.square_size), 3)
                
                # Draw the piece
                piece = self.game.board[row][col]
                if piece.type != PieceType.EMPTY:
                    if (piece.type, piece.color) in self.piece_images:
                        self.screen.blit(self.piece_images[(piece.type, piece.color)], (x, y))
        
        # Draw board border
        pygame.draw.rect(self.screen, BLACK, (self.board_offset_x, self.board_offset_y, BOARD_SIZE, BOARD_SIZE), 2)
        
        # Draw coordinates
        for i in range(8):
            # File labels (a-h)
            file_text = chr(97 + i)
            self.draw_text(file_text, self.font, BLACK, 
                          self.board_offset_x + i * self.square_size + self.square_size // 2, 
                          self.board_offset_y + BOARD_SIZE + 5, 
                          align="center")
            
            # Rank labels (1-8)
            rank_text = str(8 - i)
            self.draw_text(rank_text, self.font, BLACK, 
                          self.board_offset_x - 15, 
                          self.board_offset_y + i * self.square_size + self.square_size // 2, 
                          align="center")
    
    def handle_game_events(self, event):
        """Handle Chess-specific events"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the board was clicked
            x, y = event.pos
            
            if (self.board_offset_x <= x < self.board_offset_x + BOARD_SIZE and 
                self.board_offset_y <= y < self.board_offset_y + BOARD_SIZE):
                
                # Calculate board coordinates
                col = (x - self.board_offset_x) // self.square_size
                row = 7 - (y - self.board_offset_y) // self.square_size  # Flip y-axis
                
                if 0 <= col < 8 and 0 <= row < 8:
                    # If no square is selected, select this one
                    if self.selected_square is None:
                        piece = self.game.board[row][col]
                        if piece.type != PieceType.EMPTY and piece.color == self.game.current_player.color:
                            self.selected_square = (col, row)
                    else:
                        # If a square is already selected, try to move
                        from_col, from_row = self.selected_square
                        
                        # Check if clicking the same square (deselect)
                        if (col, row) == self.selected_square:
                            self.selected_square = None
                        else:
                            # Try to move
                            move = {
                                "from_x": from_col,
                                "from_y": from_row,
                                "to_x": col,
                                "to_y": row
                            }
                            
                            # Check for special moves
                            piece = self.game.board[from_row][from_col]
                            
                            # Promotion
                            if (piece.type == PieceType.PAWN and 
                                ((piece.color == PieceColor.WHITE and row == 7) or 
                                 (piece.color == PieceColor.BLACK and row == 0))):
                                move["promotion"] = "Q"  # Default to queen
                            
                            # Castling
                            if (piece.type == PieceType.KING and abs(col - from_col) == 2):
                                move["castling"] = True
                            
                            # En passant
                            if (piece.type == PieceType.PAWN and 
                                abs(col - from_col) == 1 and 
                                self.game.board[row][col].type == PieceType.EMPTY):
                                move["en_passant"] = True
                            
                            if self.game.is_valid_move(move):
                                self.game.make_move(move)
                                self.selected_square = None
                            else:
                                # If invalid move, try selecting the new square instead
                                piece = self.game.board[row][col]
                                if piece.type != PieceType.EMPTY and piece.color == self.game.current_player.color:
                                    self.selected_square = (col, row)
                                else:
                                    self.selected_square = None

class GoGUI(GameGUI):
    """GUI for Go game"""
    
    def __init__(self, game):
        super().__init__(game)
        
        # Go board size
        self.board_size = game.board_size
        self.cell_size = min(BOARD_SIZE // self.board_size, 30)
        
        # Board dimensions
        self.board_width = self.cell_size * self.board_size
        self.board_height = self.cell_size * self.board_size
        
        # Board offset
        self.board_offset_x = (SCREEN_WIDTH - self.board_width) // 2
        self.board_offset_y = (SCREEN_HEIGHT - self.board_height) // 2
    
    def draw_game_content(self):
        """Draw the Go game content"""
        # Draw the board
        self.draw_board()
        
        # Draw scores
        black_score = self.game.players[0].score
        white_score = self.game.players[1].score
        
        if black_score > 0 or white_score > 0:
            self.draw_text(f"Black Score: {black_score}", self.font, BLACK, MARGIN, MARGIN * 2)
            self.draw_text(f"White Score: {white_score}", self.font, BLACK, MARGIN, MARGIN * 3)
        
        # Draw captures
        black_captures = self.game.players[0].captures
        white_captures = self.game.players[1].captures
        
        self.draw_text(f"Black Captures: {black_captures}", self.font, BLACK, MARGIN, MARGIN * 4)
        self.draw_text(f"White Captures: {white_captures}", self.font, BLACK, MARGIN, MARGIN * 5)
        
        # Draw "Pass" button
        pass_button = pygame.Rect(SCREEN_WIDTH - 150, SCREEN_HEIGHT - 50, 100, 30)
        pygame.draw.rect(self.screen, GRAY, pass_button)
        pygame.draw.rect(self.screen, BLACK, pass_button, 2)
        self.draw_text("Pass", self.font, BLACK, pass_button.centerx, pass_button.centery, align="center")
        
        self.pass_button_rect = pass_button
    
    def draw_board(self):
        """Draw the Go board"""
        # Draw the board background
        pygame.draw.rect(self.screen, LIGHT_BROWN, 
                        (self.board_offset_x - self.cell_size // 2, 
                         self.board_offset_y - self.cell_size // 2,
                         self.board_width + self.cell_size,
                         self.board_height + self.cell_size))
        
        # Draw the grid lines
        for i in range(self.board_size):
            # Horizontal lines
            pygame.draw.line(self.screen, BLACK,
                           (self.board_offset_x, self.board_offset_y + i * self.cell_size),
                           (self.board_offset_x + self.board_width - self.cell_size, self.board_offset_y + i * self.cell_size))
            
            # Vertical lines
            pygame.draw.line(self.screen, BLACK,
                           (self.board_offset_x + i * self.cell_size, self.board_offset_y),
                           (self.board_offset_x + i * self.cell_size, self.board_offset_y + self.board_height - self.cell_size))
        
        # Draw star points (hoshi)
        star_points = self._get_star_points()
        for point in star_points:
            x, y = point
            center_x = self.board_offset_x + x * self.cell_size
            center_y = self.board_offset_y + y * self.cell_size
            pygame.draw.circle(self.screen, BLACK, (center_x, center_y), self.cell_size // 5)
        
        # Draw the stones
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.game.board[y][x] != Stone.EMPTY:
                    center_x = self.board_offset_x + x * self.cell_size
                    center_y = self.board_offset_y + y * self.cell_size
                    color = BLACK if self.game.board[y][x] == Stone.BLACK else WHITE
                    pygame.draw.circle(self.screen, color, (center_x, center_y), self.cell_size // 2 - 1)
                    pygame.draw.circle(self.screen, BLACK, (center_x, center_y), self.cell_size // 2 - 1, 1)
    
    def _get_star_points(self):
        """Get the positions of star points based on board size"""
        star_points = []
        
        if self.board_size == 19:
            # Traditional 19x19 board has 9 star points
            points = [3, 9, 15]
            for y in points:
                for x in points:
                    star_points.append((x, y))
        elif self.board_size == 13:
            # 13x13 board has 5 star points
            points = [3, 6, 9]
            middle = 6
            star_points.append((middle, middle))
            for point in points:
                if point != middle:
                    star_points.append((point, middle))
                    star_points.append((middle, point))
        elif self.board_size == 9:
            # 9x9 board has 5 star points
            points = [2, 4, 6]
            middle = 4
            star_points.append((middle, middle))
            for point in points:
                if point != middle:
                    star_points.append((point, middle))
                    star_points.append((middle, point))
        
        return star_points
    
    def handle_game_events(self, event):
        """Handle Go-specific events"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the pass button was clicked
            if hasattr(self, 'pass_button_rect') and self.pass_button_rect.collidepoint(event.pos):
                move = {"type": "pass"}
                self.game.make_move(move)
                return
            
            # Check if the board was clicked
            x, y = event.pos
            
            if (self.board_offset_x - self.cell_size // 2 <= x < self.board_offset_x + self.board_width + self.cell_size // 2 and 
                self.board_offset_y - self.cell_size // 2 <= y < self.board_offset_y + self.board_height + self.cell_size // 2):
                
                # Calculate the closest intersection
                board_x = round((x - self.board_offset_x) / self.cell_size)
                board_y = round((y - self.board_offset_y) / self.cell_size)
                
                if 0 <= board_x < self.board_size and 0 <= board_y < self.board_size:
                    # Try to place a stone
                    move = {"type": "place", "x": board_x, "y": board_y}
                    if self.game.is_valid_move(move):
                        self.game.make_move(move)

def main():
    """Main entry point for the pygame UI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Games Kernel Pygame UI")
    parser.add_argument("game", choices=["mahjong", "chess", "go"], help="Game to play")
    parser.add_argument("--board-size", "-b", type=int, default=19, help="Board size for Go (default: 19)")
    
    args = parser.parse_args()
    
    # Create game engine
    engine = GameEngine()
    engine.register_game(MahjongGame)
    engine.register_game(ChessGame)
    engine.register_game(GoGame)
    
    # Create game instance
    if args.game == "mahjong":
        winds = [Wind.EAST, Wind.SOUTH, Wind.WEST, Wind.NORTH]
        players = [MahjongPlayer(f"Player {i+1}", wind) for i, wind in enumerate(winds)]
        game = MahjongGame(players)
        gui = MahjongGUI(game)
    elif args.game == "chess":
        players = [
            ChessPlayer("White Player", PieceColor.WHITE),
            ChessPlayer("Black Player", PieceColor.BLACK)
        ]
        game = ChessGame(players)
        gui = ChessGUI(game)
    elif args.game == "go":
        players = [
            GoPlayer("Black Player", Stone.BLACK),
            GoPlayer("White Player", Stone.WHITE)
        ]
        game = GoGame(players, board_size=args.board_size)
        gui = GoGUI(game)
    
    game.initialize_game()
    gui.run()

if __name__ == "__main__":
    main()