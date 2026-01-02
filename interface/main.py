"""
Image Processing Application
University of Kurdistan - Computer Department
Built with CustomTkinter
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
from PIL import Image
import numpy as np
from typing import Optional, List
import threading

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

        # Operations frame
        self.operations_frame = ctk.CTkFrame(self.sidebar)
        self.operations_frame.pack(fill="x", pady=5)

        # --- Operation 1: Reduce Resolution ---
        self.btn_reduce_res = ctk.CTkButton(
            self.operations_frame,
            text="1. Reduce Resolution (1/2)",
            command=self.op_reduce_resolution,
            height=35
        )
        self.btn_reduce_res.pack(fill="x", padx=10, pady=5)

        # --- Operation 2: Negative Image ---
        self.btn_negative = ctk.CTkButton(
            self.operations_frame,
            text="2. Negative Image",
            command=self.op_negative,
            height=35
        )
        self.btn_negative.pack(fill="x", padx=10, pady=5)

        # --- Operation 3: Log Transformation ---
        self.btn_log = ctk.CTkButton(
            self.operations_frame,
            text="3. Log Transformation",
            command=self.op_log_transform,
            height=35
        )
        self.btn_log.pack(fill="x", padx=10, pady=5)

        # --- Operation 4: Gamma Transformation ---
        gamma_frame = ctk.CTkFrame(self.operations_frame)
        gamma_frame.pack(fill="x", padx=10, pady=5)

        self.btn_gamma = ctk.CTkButton(
            gamma_frame,
            text="4. Gamma",
            command=self.op_gamma_transform,
            height=35,
            width=120
        )
        self.btn_gamma.pack(side="left", padx=(0, 5))

        self.gamma_slider = ctk.CTkSlider(
            gamma_frame,
            from_=0.1,
            to=3.0,
            number_of_steps=29,
            width=100
        )
        self.gamma_slider.set(1.0)
        self.gamma_slider.pack(side="left", padx=5)

        self.gamma_label = ctk.CTkLabel(gamma_frame, text="1.0", width=30)
        self.gamma_label.pack(side="left")
        self.gamma_slider.configure(command=self._update_gamma_label)

        # --- Operation 5: Show Histogram ---
        self.btn_histogram = ctk.CTkButton(
            self.operations_frame,
            text="5. Show Histogram",
            command=self.op_show_histogram,
            height=35
        )
        self.btn_histogram.pack(fill="x", padx=10, pady=5)

        # --- Operation 6: Histogram Equalization ---
        self.btn_hist_eq = ctk.CTkButton(
            self.operations_frame,
            text="6. Histogram Equalization",
            command=self.op_histogram_equalization,
            height=35
        )
        self.btn_hist_eq.pack(fill="x", padx=10, pady=5)

        # --- Operation 7: Blur Filter ---
        blur_frame = ctk.CTkFrame(self.operations_frame)
        blur_frame.pack(fill="x", padx=10, pady=5)

        self.btn_blur = ctk.CTkButton(
            blur_frame,
            text="7. Blur",
            command=self.op_blur,
            height=35,
            width=120
        )
        self.btn_blur.pack(side="left", padx=(0, 5))

        self.blur_size = ctk.CTkOptionMenu(
            blur_frame,
            values=["3", "5", "7", "9", "11"],
            width=70
        )
        self.blur_size.set("3")
        self.blur_size.pack(side="left")

        # --- Operation 8: Sharpen Filter ---
        self.btn_sharpen = ctk.CTkButton(
            self.operations_frame,
            text="8. Sharpen (3x3)",
            command=self.op_sharpen,
            height=35
        )
        self.btn_sharpen.pack(fill="x", padx=10, pady=5)

        # --- Operation 9: Gradient Magnitude ---
        self.btn_gradient = ctk.CTkButton(
            self.operations_frame,
            text="9. Gradient Magnitude",
            command=self.op_gradient_magnitude,
            height=35
        )
        self.btn_gradient.pack(fill="x", padx=10, pady=5)

        # --- Operation 10: Edge Detection ---
        edge_frame = ctk.CTkFrame(self.operations_frame)
        edge_frame.pack(fill="x", padx=10, pady=5)

        self.btn_edge = ctk.CTkButton(
            edge_frame,
            text="10. Edge",
            command=self.op_edge_detection,
            height=35,
            width=120
        )
        self.btn_edge.pack(side="left", padx=(0, 5))

        self.edge_threshold = ctk.CTkSlider(
            edge_frame,
            from_=10,
            to=200,
            number_of_steps=19,
            width=100
        )
        self.edge_threshold.set(50)
        self.edge_threshold.pack(side="left", padx=5)

        self.edge_label = ctk.CTkLabel(edge_frame, text="50", width=30)
        self.edge_label.pack(side="left")
        self.edge_threshold.configure(command=self._update_edge_label)

        # Separator for Noise operations
        separator2 = ctk.CTkFrame(self.operations_frame, height=2, fg_color="gray50")
        separator2.pack(fill="x", pady=10, padx=10)

        noise_label = ctk.CTkLabel(
            self.operations_frame,
            text="Noise Operations",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        noise_label.pack(pady=5)

        # --- Add Salt & Pepper Noise ---
        self.btn_add_sp_noise = ctk.CTkButton(
            self.operations_frame,
            text="Add Salt & Pepper Noise",
            command=self.op_add_salt_pepper,
            height=35,
            fg_color="#8E44AD",
            hover_color="#7D3C98"
        )
        self.btn_add_sp_noise.pack(fill="x", padx=10, pady=5)

        # --- Add Gaussian Noise ---
        self.btn_add_gauss_noise = ctk.CTkButton(
            self.operations_frame,
            text="Add Gaussian Noise",
            command=self.op_add_gaussian_noise,
            height=35,
            fg_color="#8E44AD",
            hover_color="#7D3C98"
        )
        self.btn_add_gauss_noise.pack(fill="x", padx=10, pady=5)

        # --- Median Filter ---
        self.btn_median = ctk.CTkButton(
            self.operations_frame,
            text="Median Filter (5x5)",
            command=self.op_median_filter,
            height=35,
            fg_color="#27AE60",
            hover_color="#1E8449"
        )
        self.btn_median.pack(fill="x", padx=10, pady=5)

        # --- Gaussian Filter ---
        self.btn_gauss_filter = ctk.CTkButton(
            self.operations_frame,
            text="Gaussian Filter (5x5)",
            command=self.op_gaussian_filter,
            height=35,
            fg_color="#27AE60",
            hover_color="#1E8449"
        )
        self.btn_gauss_filter.pack(fill="x", padx=10, pady=5)

    def _update_gamma_label(self, value):
        """Update gamma label when slider changes"""
        self.gamma_label.configure(text=f"{value:.1f}")

    def _update_edge_label(self, value):
        """Update edge threshold label when slider changes"""
        self.edge_label.configure(text=f"{int(value)}")

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

        # Container for image and histogram
        self.display_container = ctk.CTkFrame(self.main_frame)
        self.display_container.pack(fill="both", expand=True)

        # Image canvas frame (left side)
        self.canvas_frame = ctk.CTkFrame(self.display_container)
        self.canvas_frame.pack(side="left", fill="both", expand=True)

        # Image label
        self.image_label = ctk.CTkLabel(
            self.canvas_frame,
            text="Load an image to start",
            font=ctk.CTkFont(size=16)
        )
        self.image_label.pack(expand=True)

        # Histogram frame (right side)
        self.histogram_frame = ctk.CTkFrame(self.display_container, width=300)
        self.histogram_frame.pack(side="right", fill="y", padx=(10, 0))
        self.histogram_frame.pack_propagate(False)

        # Histogram title
        hist_title = ctk.CTkLabel(
            self.histogram_frame,
            text="Histogram",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        hist_title.pack(pady=10)

        # Histogram canvas
        self.hist_canvas = tk.Canvas(
            self.histogram_frame,
            bg="#2b2b2b",
            width=280,
            height=200,
            highlightthickness=0
        )
        self.hist_canvas.pack(padx=10, pady=5)

        # Histogram stats
        self.hist_stats_label = ctk.CTkLabel(
            self.histogram_frame,
            text="",
            font=ctk.CTkFont(size=11),
            justify="left"
        )
        self.hist_stats_label.pack(pady=10, padx=10, anchor="w")

        # Store CTkImage reference
        self.ctk_image = None

    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = ctk.CTkFrame(self, height=40)
        self.status_bar.pack(side="bottom", fill="x")

        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="Ready",
            font=ctk.CTkFont(size=11)
        )
        self.status_label.pack(side="left", padx=10)

        # Progress bar (hidden by default)
        self.progress_frame = ctk.CTkFrame(self.status_bar)
        self.progress_frame.pack(side="left", padx=20)

        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="Processing...",
            font=ctk.CTkFont(size=11)
        )
        self.progress_label.pack(side="left", padx=(0, 10))

        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame,
            width=200,
            mode="indeterminate"
        )
        self.progress_bar.pack(side="left")
        self.progress_frame.pack_forget()  # Hide initially

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

        # Create CTkImage (handles HighDPI automatically)
        self.ctk_image = ctk.CTkImage(
            light_image=img,
            dark_image=img,
            size=new_size
        )

        # Update label
        self.image_label.configure(image=self.ctk_image, text="")

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
            # Update histogram
            self._update_histogram()
        else:
            self.info_label.configure(text="No image loaded")

    def _update_histogram(self):
        """Update histogram display in main window"""
        if self.current_image is None:
            return

        # Clear previous histogram
        self.hist_canvas.delete("all")

        # Calculate histogram
        img_uint8 = self.current_image.astype(np.uint8)
        histogram = np.zeros(256)
        for val in img_uint8.flatten():
            histogram[val] += 1

        # Draw histogram
        max_val = histogram.max() if histogram.max() > 0 else 1
        canvas_width = 280
        canvas_height = 200
        bar_width = canvas_width / 256

        for i in range(256):
            if histogram[i] > 0:
                height = (histogram[i] / max_val) * (canvas_height - 20)
                x0 = i * bar_width
                x1 = x0 + bar_width
                # Color gradient from dark to light
                gray_val = i
                color = f"#{gray_val:02x}{gray_val:02x}{gray_val:02x}"
                self.hist_canvas.create_rectangle(
                    x0, canvas_height - height,
                    x1, canvas_height,
                    fill=color, outline=""
                )

        # Draw axis lines
        self.hist_canvas.create_line(0, canvas_height - 1, canvas_width, canvas_height - 1, fill="#555555")
        self.hist_canvas.create_text(5, 5, text="0", fill="#888888", anchor="nw", font=("Arial", 8))
        self.hist_canvas.create_text(canvas_width - 5, 5, text="255", fill="#888888", anchor="ne", font=("Arial", 8))

        # Update stats
        self.hist_stats_label.configure(
            text=f"Min: {int(self.current_image.min())}\n"
                 f"Max: {int(self.current_image.max())}\n"
                 f"Mean: {self.current_image.mean():.1f}\n"
                 f"Std: {self.current_image.std():.1f}"
        )

    def _show_progress(self, message="Processing..."):
        """Show progress bar"""
        self.progress_label.configure(text=message)
        self.progress_frame.pack(side="left", padx=20)
        self.progress_bar.start()
        self.update()

    def _hide_progress(self):
        """Hide progress bar"""
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
        self.update()

    def apply_operation(self, operation_func, *args, **kwargs):
        """Apply an operation to the current image with progress indicator"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        # Show progress bar
        self._show_progress("Processing...")

        def run_operation():
            try:
                self._add_to_history()
                self.current_image = operation_func(self.current_image, *args, **kwargs)
                # Update UI in main thread
                self.after(0, self._operation_complete)
            except Exception as e:
                self.after(0, lambda: self._operation_failed(str(e)))

        # Run in thread for heavy operations
        thread = threading.Thread(target=run_operation)
        thread.start()

    def _operation_complete(self):
        """Called when operation completes successfully"""
        self._hide_progress()
        self._display_image()
        self._update_buttons_state()
        self._update_info()

    def _operation_failed(self, error_msg):
        """Called when operation fails"""
        self._hide_progress()
        if self.history:
            self.history.pop()  # Remove failed state from history
        messagebox.showerror("Error", f"Operation failed:\n{error_msg}")

    # ========================================
    # Image Processing Operations
    # ========================================

    def op_reduce_resolution(self):
        """Operation 1: Reduce resolution by half using subsampling"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        def reduce_resolution(img):
            # Take every other pixel (subsampling)
            return img[::2, ::2].copy()

        self.apply_operation(reduce_resolution)
        self.status_label.configure(text="Applied: Reduce Resolution (1/2)")

    def op_negative(self):
        """Operation 2: Create negative image (s = 255 - r)"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        def negative(img):
            return 255 - img

        self.apply_operation(negative)
        self.status_label.configure(text="Applied: Negative Image (s = 255 - r)")

    def op_log_transform(self):
        """Operation 3: Log transformation (y = c * log(1 + x))"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        def log_transform(img):
            # c = 255 / log(256) to scale output to [0, 255]
            c = 255 / np.log(1 + 255)
            return c * np.log(1 + img)

        self.apply_operation(log_transform)
        self.status_label.configure(text="Applied: Log Transformation")

    def op_gamma_transform(self):
        """Operation 4: Gamma/Power transformation (y = c * x^gamma)"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        gamma = self.gamma_slider.get()

        def gamma_transform(img):
            # Normalize to [0, 1], apply gamma, scale back to [0, 255]
            normalized = img / 255.0
            return np.power(normalized, gamma) * 255

        self.apply_operation(gamma_transform)
        self.status_label.configure(text=f"Applied: Gamma Transformation (γ = {gamma:.1f})")

    def op_show_histogram(self):
        """Operation 5: Toggle histogram panel visibility"""
        if self.histogram_frame.winfo_ismapped():
            self.histogram_frame.pack_forget()
            self.btn_histogram.configure(text="5. Show Histogram")
            self.status_label.configure(text="Histogram hidden")
        else:
            self.histogram_frame.pack(side="right", fill="y", padx=(10, 0))
            self.btn_histogram.configure(text="5. Hide Histogram")
            if self.current_image is not None:
                self._update_histogram()
            self.status_label.configure(text="Histogram shown")

    def op_histogram_equalization(self):
        """Operation 6: Histogram equalization"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        def histogram_equalization(img):
            M, N = img.shape

            # Calculate histogram
            histogram = np.zeros(256)
            img_uint8 = img.astype(np.uint8)
            for i in range(M):
                for j in range(N):
                    histogram[img_uint8[i, j]] += 1

            # Calculate CDF
            cdf = np.zeros(256)
            cdf[0] = histogram[0]
            for i in range(1, 256):
                cdf[i] = cdf[i-1] + histogram[i]

            # Normalize CDF
            cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
            transfer_function = cdf_normalized.astype(np.uint8)

            # Apply transfer function
            result = np.zeros_like(img)
            for i in range(M):
                for j in range(N):
                    result[i, j] = transfer_function[img_uint8[i, j]]

            return result

        self.apply_operation(histogram_equalization)
        self.status_label.configure(text="Applied: Histogram Equalization")

    def op_blur(self):
        """Operation 7: Blur filter (NxN averaging)"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        n_filter = int(self.blur_size.get())

        def blur_filter(img):
            M, N = img.shape
            pad = n_filter // 2

            # Zero padding
            img_padded = np.zeros((M + 2*pad, N + 2*pad))
            img_padded[pad:pad+M, pad:pad+N] = img

            # Apply averaging filter
            result = np.zeros_like(img)
            for i in range(M):
                for j in range(N):
                    neighborhood = img_padded[i:i+n_filter, j:j+n_filter]
                    result[i, j] = np.mean(neighborhood)

            return result

        self.apply_operation(blur_filter)
        self.status_label.configure(text=f"Applied: Blur Filter ({n_filter}x{n_filter})")

    def op_sharpen(self):
        """Operation 8: Sharpen filter (3x3 Laplacian-based)"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        def sharpen_filter(img):
            M, N = img.shape

            # Laplacian-based sharpening kernel
            kernel = np.array([[ 0, -1,  0],
                               [-1,  5, -1],
                               [ 0, -1,  0]])

            pad = 1
            img_padded = np.zeros((M + 2*pad, N + 2*pad))
            img_padded[pad:pad+M, pad:pad+N] = img

            result = np.zeros_like(img)
            for i in range(M):
                for j in range(N):
                    neighborhood = img_padded[i:i+3, j:j+3]
                    value = np.sum(neighborhood * kernel)
                    result[i, j] = np.clip(value, 0, 255)

            return result

        self.apply_operation(sharpen_filter)
        self.status_label.configure(text="Applied: Sharpen Filter (3x3)")

    def op_gradient_magnitude(self):
        """Operation 9: Gradient magnitude using Sobel operators"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        def gradient_magnitude(img):
            M, N = img.shape

            # Sobel kernels
            sobel_x = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])

            sobel_y = np.array([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]])

            pad = 1
            img_padded = np.zeros((M + 2*pad, N + 2*pad))
            img_padded[pad:pad+M, pad:pad+N] = img

            gradient_x = np.zeros_like(img)
            gradient_y = np.zeros_like(img)

            for i in range(M):
                for j in range(N):
                    neighborhood = img_padded[i:i+3, j:j+3]
                    gradient_x[i, j] = np.sum(neighborhood * sobel_x)
                    gradient_y[i, j] = np.sum(neighborhood * sobel_y)

            # Magnitude
            magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            # Normalize to [0, 255]
            magnitude = (magnitude / magnitude.max()) * 255

            return magnitude

        self.apply_operation(gradient_magnitude)
        self.status_label.configure(text="Applied: Gradient Magnitude (Sobel)")

    def op_edge_detection(self):
        """Operation 10: Edge detection with threshold"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        threshold = int(self.edge_threshold.get())

        def edge_detection(img):
            M, N = img.shape

            # Sobel kernels
            sobel_x = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])

            sobel_y = np.array([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]])

            pad = 1
            img_padded = np.zeros((M + 2*pad, N + 2*pad))
            img_padded[pad:pad+M, pad:pad+N] = img

            gradient_x = np.zeros_like(img)
            gradient_y = np.zeros_like(img)

            for i in range(M):
                for j in range(N):
                    neighborhood = img_padded[i:i+3, j:j+3]
                    gradient_x[i, j] = np.sum(neighborhood * sobel_x)
                    gradient_y[i, j] = np.sum(neighborhood * sobel_y)

            # Magnitude and normalize
            magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            magnitude = (magnitude / magnitude.max()) * 255

            # Apply threshold
            edges = np.zeros_like(img)
            edges[magnitude > threshold] = 255

            return edges

        self.apply_operation(edge_detection)
        self.status_label.configure(text=f"Applied: Edge Detection (threshold={threshold})")

    def op_add_salt_pepper(self):
        """Add Salt & Pepper noise (15%)"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        def add_salt_pepper(img, amount=0.15):
            M, N = img.shape
            noisy = img.copy()
            total_pixels = M * N

            # Add Salt (white pixels)
            num_salt = int(total_pixels * amount / 2)
            for _ in range(num_salt):
                i = np.random.randint(0, M)
                j = np.random.randint(0, N)
                noisy[i, j] = 255

            # Add Pepper (black pixels)
            num_pepper = int(total_pixels * amount / 2)
            for _ in range(num_pepper):
                i = np.random.randint(0, M)
                j = np.random.randint(0, N)
                noisy[i, j] = 0

            return noisy

        self.apply_operation(add_salt_pepper)
        self.status_label.configure(text="Applied: Salt & Pepper Noise (15%)")

    def op_add_gaussian_noise(self):
        """Add Gaussian noise"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        def add_gaussian_noise(img, mean=0, std=50):
            noise = np.random.normal(mean, std, img.shape)
            noisy = img + noise
            return np.clip(noisy, 0, 255)

        self.apply_operation(add_gaussian_noise)
        self.status_label.configure(text="Applied: Gaussian Noise (σ=50)")

    def op_median_filter(self):
        """Median filter (5x5) - good for Salt & Pepper noise"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        def median_filter(img, size=5):
            M, N = img.shape
            pad = size // 2

            img_padded = np.zeros((M + 2*pad, N + 2*pad))
            img_padded[pad:pad+M, pad:pad+N] = img

            result = np.zeros_like(img)
            for i in range(M):
                for j in range(N):
                    neighborhood = img_padded[i:i+size, j:j+size]
                    result[i, j] = np.median(neighborhood)

            return result

        self.apply_operation(median_filter)
        self.status_label.configure(text="Applied: Median Filter (5x5)")

    def op_gaussian_filter(self):
        """Gaussian filter (5x5) - good for Gaussian noise"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        def gaussian_filter(img, size=5, sigma=1.5):
            M, N = img.shape
            pad = size // 2

            # Create Gaussian kernel
            kernel = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    x = i - pad
                    y = j - pad
                    kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()

            # Apply filter
            img_padded = np.zeros((M + 2*pad, N + 2*pad))
            img_padded[pad:pad+M, pad:pad+N] = img

            result = np.zeros_like(img)
            for i in range(M):
                for j in range(N):
                    neighborhood = img_padded[i:i+size, j:j+size]
                    result[i, j] = np.sum(neighborhood * kernel)

            return result

        self.apply_operation(gaussian_filter)
        self.status_label.configure(text="Applied: Gaussian Filter (5x5, σ=1.5)")


if __name__ == "__main__":
    app = ImageProcessorApp()
    app.mainloop()
