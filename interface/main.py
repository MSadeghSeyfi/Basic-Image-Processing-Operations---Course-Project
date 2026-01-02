"""
Image Processing Application
University of Kurdistan - Computer Department
Built with CustomTkinter
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from typing import Optional, List

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ImageProcessorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Image Processing - University of Kurdistan")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        # Image data
        self.original_image: Optional[np.ndarray] = None
        self.current_image: Optional[np.ndarray] = None
        self.history: List[np.ndarray] = []
        self.max_history = 20

        # Build UI
        self._create_layout()
        self._create_toolbar()
        self._create_image_display()
        self._create_status_bar()

    def _create_layout(self):
        """Create main layout frames"""
        # Left sidebar for operations
        self.sidebar = ctk.CTkScrollableFrame(self, width=250)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        # Main content area
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    def _create_toolbar(self):
        """Create toolbar with file operations"""
        # Title
        title = ctk.CTkLabel(
            self.sidebar,
            text="Image Processing",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=(10, 20))

        # File Operations Frame
        file_frame = ctk.CTkFrame(self.sidebar)
        file_frame.pack(fill="x", pady=10)

        file_label = ctk.CTkLabel(
            file_frame,
            text="File Operations",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        file_label.pack(pady=5)

        # Load button
        self.btn_load = ctk.CTkButton(
            file_frame,
            text="Load Image",
            command=self.load_image,
            height=40
        )
        self.btn_load.pack(fill="x", padx=10, pady=5)

        # Save button
        self.btn_save = ctk.CTkButton(
            file_frame,
            text="Save Image",
            command=self.save_image,
            height=40,
            state="disabled"
        )
        self.btn_save.pack(fill="x", padx=10, pady=5)

        # Undo button
        self.btn_undo = ctk.CTkButton(
            file_frame,
            text="Undo",
            command=self.undo,
            height=40,
            state="disabled",
            fg_color="#E67E22",
            hover_color="#D35400"
        )
        self.btn_undo.pack(fill="x", padx=10, pady=5)

        # Reset button
        self.btn_reset = ctk.CTkButton(
            file_frame,
            text="Reset to Original",
            command=self.reset_image,
            height=40,
            state="disabled",
            fg_color="#E74C3C",
            hover_color="#C0392B"
        )
        self.btn_reset.pack(fill="x", padx=10, pady=5)

        # Separator
        separator = ctk.CTkFrame(self.sidebar, height=2, fg_color="gray50")
        separator.pack(fill="x", pady=15)

        # Operations label (placeholder for future operations)
        ops_label = ctk.CTkLabel(
            self.sidebar,
            text="Operations",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        ops_label.pack(pady=5)

        # Operations frame (will add buttons here later)
        self.operations_frame = ctk.CTkFrame(self.sidebar)
        self.operations_frame.pack(fill="x", pady=5)

    def _create_image_display(self):
        """Create image display area"""
        # Image info frame
        self.info_frame = ctk.CTkFrame(self.main_frame, height=40)
        self.info_frame.pack(fill="x", pady=(0, 10))

        self.info_label = ctk.CTkLabel(
            self.info_frame,
            text="No image loaded",
            font=ctk.CTkFont(size=12)
        )
        self.info_label.pack(pady=10)

        # Image canvas frame
        self.canvas_frame = ctk.CTkFrame(self.main_frame)
        self.canvas_frame.pack(fill="both", expand=True)

        # Image label
        self.image_label = ctk.CTkLabel(
            self.canvas_frame,
            text="Load an image to start",
            font=ctk.CTkFont(size=16)
        )
        self.image_label.pack(expand=True)

        # Store PhotoImage reference
        self.photo_image = None

    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = ctk.CTkFrame(self, height=30)
        self.status_bar.pack(side="bottom", fill="x")

        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="Ready",
            font=ctk.CTkFont(size=11)
        )
        self.status_label.pack(side="left", padx=10)

        self.history_label = ctk.CTkLabel(
            self.status_bar,
            text="History: 0",
            font=ctk.CTkFont(size=11)
        )
        self.history_label.pack(side="right", padx=10)

    def load_image(self):
        """Load image from file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                # Load image
                img = Image.open(file_path)

                # Convert to grayscale
                if img.mode != 'L':
                    img = img.convert('L')

                # Convert to numpy array
                self.original_image = np.array(img, dtype=np.float64)
                self.current_image = self.original_image.copy()
                self.history = []

                # Update display
                self._display_image()
                self._update_buttons_state()
                self._update_info()
                self.status_label.configure(text=f"Loaded: {file_path.split('/')[-1]}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")

    def save_image(self):
        """Save current image to file"""
        if self.current_image is None:
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("BMP files", "*.bmp"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                img = Image.fromarray(self.current_image.astype(np.uint8))
                img.save(file_path)
                self.status_label.configure(text=f"Saved: {file_path.split('/')[-1]}")
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")

    def undo(self):
        """Undo last operation"""
        if self.history:
            self.current_image = self.history.pop()
            self._display_image()
            self._update_buttons_state()
            self._update_info()
            self.status_label.configure(text="Undo successful")

    def reset_image(self):
        """Reset to original image"""
        if self.original_image is not None:
            self._add_to_history()
            self.current_image = self.original_image.copy()
            self._display_image()
            self._update_buttons_state()
            self._update_info()
            self.status_label.configure(text="Reset to original")

    def _add_to_history(self):
        """Add current state to history"""
        if self.current_image is not None:
            self.history.append(self.current_image.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
            self._update_buttons_state()

    def _display_image(self):
        """Display current image"""
        if self.current_image is None:
            return

        # Get display area size
        self.canvas_frame.update()
        max_width = self.canvas_frame.winfo_width() - 40
        max_height = self.canvas_frame.winfo_height() - 40

        if max_width <= 0 or max_height <= 0:
            max_width = 800
            max_height = 600

        # Create PIL image
        img = Image.fromarray(self.current_image.astype(np.uint8))

        # Calculate resize ratio
        ratio = min(max_width / img.width, max_height / img.height, 1.0)
        new_size = (int(img.width * ratio), int(img.height * ratio))

        # Resize for display
        img_display = img.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to PhotoImage
        self.photo_image = ImageTk.PhotoImage(img_display)

        # Update label
        self.image_label.configure(image=self.photo_image, text="")

    def _update_buttons_state(self):
        """Update button states based on current state"""
        has_image = self.current_image is not None
        has_history = len(self.history) > 0

        self.btn_save.configure(state="normal" if has_image else "disabled")
        self.btn_undo.configure(state="normal" if has_history else "disabled")
        self.btn_reset.configure(state="normal" if has_image else "disabled")

        self.history_label.configure(text=f"History: {len(self.history)}")

    def _update_info(self):
        """Update image info display"""
        if self.current_image is not None:
            h, w = self.current_image.shape
            self.info_label.configure(
                text=f"Size: {w} x {h} | "
                     f"Min: {self.current_image.min():.0f} | "
                     f"Max: {self.current_image.max():.0f} | "
                     f"Mean: {self.current_image.mean():.1f}"
            )
        else:
            self.info_label.configure(text="No image loaded")

    def apply_operation(self, operation_func, *args, **kwargs):
        """Apply an operation to the current image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        try:
            self._add_to_history()
            self.current_image = operation_func(self.current_image, *args, **kwargs)
            self._display_image()
            self._update_buttons_state()
            self._update_info()
        except Exception as e:
            self.history.pop()  # Remove failed state from history
            messagebox.showerror("Error", f"Operation failed:\n{str(e)}")


if __name__ == "__main__":
    app = ImageProcessorApp()
    app.mainloop()
