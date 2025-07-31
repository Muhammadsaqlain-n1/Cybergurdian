import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import os


class ToxicityGUI:
    def __init__(self):
        self.data = []
        self.current_file = None
        self.load_existing_components()
        self.setup_attractive_gui()

    def load_existing_components(self):
        try:
            file = open(r"C:\Users\saqla\Downloads\stopwords.txt", 'r')
            content = file.read()
            self.content_list = content.split('\n')
            file.close()

            self.trained_model = pickle.load(open('LinearSVC.pkl', 'rb'))
            self.vocab = pickle.load(open("tfidfVectorizer.pkl", "rb"))

            print("Models loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load models: {e}")

    def remove_pattern(self, input_txt, pattern):
        if (type(input_txt) == str):
            r = re.findall(pattern, input_txt)
            for i in r:
                input_txt = re.sub(re.escape(i), '', input_txt)
            return input_txt
        else:
            return ''

    def preprocess_text(self, text):
        processed_text = self.remove_pattern(text, '@[\w]*')
        processed_text = re.sub('[^a-zA-Z#]', ' ', processed_text)
        processed_text = ' '.join([w for w in processed_text.split() if len(w) > 3])

        try:
            lemmatizer = nltk.stem.WordNetLemmatizer()
            words = processed_text.split()
            words = [lemmatizer.lemmatize(i) for i in words]
            processed_text = ' '.join(words)
        except:
            pass

        return processed_text

    def predict_toxicity(self, text):
        try:
            processed_text = self.preprocess_text(text)

            tfidf_vector = TfidfVectorizer(
                stop_words=self.content_list,
                lowercase=True,
                vocabulary=self.vocab
            )

            preprocessed_data = tfidf_vector.fit_transform([processed_text])
            prediction = self.trained_model.predict(preprocessed_data)

            if prediction[0] == 1:
                return "Bullying"
            else:
                return "Non-Bullying"

        except Exception as e:
            return f"Error: {e}"

    def create_modern_button(self, parent, text, command, bg_color, hover_color, text_color="white", width=20,
                             height=2):
        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 11, "bold"),
            bg=bg_color,
            fg=text_color,
            relief="flat",
            bd=0,
            cursor="hand2",
            width=width,
            height=height,
            activebackground=hover_color,
            activeforeground=text_color
        )

        def on_enter(e):
            button.config(bg=hover_color)

        def on_leave(e):
            button.config(bg=bg_color)

        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

        return button

    def setup_attractive_gui(self):
        self.root = tk.Tk()
        self.root.title("🛡️ Advanced Toxicity Detection System - Light Edition")
        self.root.geometry("950x800")
        self.root.configure(bg='#f8fafc')
        self.root.resizable(True, True)

        self.setup_styles()

        self.main_frame = tk.Frame(self.root, bg='#f8fafc')
        self.main_frame.pack(fill='both', expand=True, padx=15, pady=15)

        self.create_header()
        self.create_file_upload_section()
        self.create_input_section()
        self.create_control_buttons()
        self.create_results_section()
        self.create_footer()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('Modern.TFrame', background='#ffffff')
        style.configure('Modern.TLabel', background='#ffffff', foreground='#1f2937')

    def create_header(self):
        header_frame = tk.Frame(self.main_frame, bg='#e0f2fe', height=120, relief='raised', bd=3)
        header_frame.pack(fill='x', pady=(0, 20))
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="🛡️ ADVANCED TOXICITY DETECTION",
            font=("Segoe UI", 26, "bold"),
            bg='#e0f2fe',
            fg='#0369a1',
            pady=15
        )
        title_label.pack()

        subtitle_label = tk.Label(
            header_frame,
            text="AI-Powered Bullying Detection • Text Input & File Upload",
            font=("Segoe UI", 14, "italic"),
            bg='#e0f2fe',
            fg='#7c3aed'
        )
        subtitle_label.pack()

        status_frame = tk.Frame(header_frame, bg='#e0f2fe')
        status_frame.pack(pady=(10, 0))

        status_dot = tk.Label(
            status_frame,
            text="●",
            font=("Segoe UI", 18),
            bg='#e0f2fe',
            fg='#10b981'
        )
        status_dot.pack(side='left')

        status_text = tk.Label(
            status_frame,
            text="System Online & Ready",
            font=("Segoe UI", 12, "bold"),
            bg='#e0f2fe',
            fg='#10b981'
        )
        status_text.pack(side='left', padx=(8, 0))

    def create_file_upload_section(self):
        upload_container = tk.Frame(self.main_frame, bg='#f8fafc')
        upload_container.pack(fill='x', pady=(0, 20))

        upload_frame = tk.Frame(upload_container, bg='#fef3c7', relief='raised', bd=3)
        upload_frame.pack(fill='x', padx=25, pady=8)

        upload_header = tk.Frame(upload_frame, bg='#fef3c7')
        upload_header.pack(fill='x', padx=25, pady=(20, 15))

        upload_icon = tk.Label(
            upload_header,
            text="📁",
            font=("Segoe UI", 16),
            bg='#fef3c7',
            fg='#d97706'
        )
        upload_icon.pack(side='left')

        upload_label = tk.Label(
            upload_header,
            text="Upload Text File for Bullying Analysis:",
            font=("Segoe UI", 16, "bold"),
            bg='#fef3c7',
            fg='#92400e'
        )
        upload_label.pack(side='left', padx=(12, 0))

        file_select_frame = tk.Frame(upload_frame, bg='#fef3c7')
        file_select_frame.pack(fill='x', padx=25, pady=(0, 20))

        self.file_info_var = tk.StringVar(value="No file selected")
        file_info_label = tk.Label(
            file_select_frame,
            textvariable=self.file_info_var,
            font=("Segoe UI", 11),
            bg='#ffffff',
            fg='#374151',
            relief='sunken',
            bd=2,
            anchor='w',
            padx=18,
            pady=12
        )
        file_info_label.pack(fill='x', pady=(0, 15))

        button_frame = tk.Frame(file_select_frame, bg='#fef3c7')
        button_frame.pack(fill='x')

        upload_button = self.create_modern_button(
            button_frame,
            "📂 SELECT FILE",
            self.upload_file,
            "#a855f7",
            "#c084fc",
            text_color="white",
            width=18,
            height=2
        )
        upload_button.pack(side='left', padx=(0, 15))

        self.analyze_file_button = self.create_modern_button(
            button_frame,
            "🔍 ANALYZE FILE",
            self.analyze_file,
            "#06b6d4",
            "#22d3ee",
            text_color="white",
            width=18,
            height=2
        )
        self.analyze_file_button.pack(side='left', padx=15)
        self.analyze_file_button.config(state='disabled')

        clear_file_button = self.create_modern_button(
            button_frame,
            "🗑️ CLEAR FILE",
            self.clear_file,
            "#f97316",
            "#fb923c",
            text_color="white",
            width=15,
            height=2
        )
        clear_file_button.pack(side='left', padx=15)

    def create_input_section(self):
        input_container = tk.Frame(self.main_frame, bg='#f8fafc')
        input_container.pack(fill='both', expand=True, pady=(0, 20))

        input_frame = tk.Frame(input_container, bg='#f0f9ff', relief='raised', bd=3)
        input_frame.pack(fill='both', expand=True, padx=25, pady=8)

        input_label_frame = tk.Frame(input_frame, bg='#f0f9ff')
        input_label_frame.pack(fill='x', padx=25, pady=(20, 15))

        input_icon = tk.Label(
            input_label_frame,
            text="📝",
            font=("Segoe UI", 16),
            bg='#f0f9ff',
            fg='#0ea5e9'
        )
        input_icon.pack(side='left')

        input_label = tk.Label(
            input_label_frame,
            text="Or enter text manually:",
            font=("Segoe UI", 16, "bold"),
            bg='#f0f9ff',
            fg='#0369a1'
        )
        input_label.pack(side='left', padx=(12, 0))

        text_frame = tk.Frame(input_frame, bg='#f0f9ff')
        text_frame.pack(fill='both', expand=True, padx=25, pady=(0, 20))

        self.text_input = scrolledtext.ScrolledText(
            text_frame,
            height=6,
            font=("Consolas", 12),
            wrap=tk.WORD,
            bg='#ffffff',
            fg='#374151',
            insertbackground='#3b82f6',
            selectbackground='#bfdbfe',
            selectforeground='#1e40af',
            relief='sunken',
            bd=3,
            padx=18,
            pady=18
        )
        self.text_input.pack(fill='both', expand=True)

        placeholder_text = "Type or paste your text here for manual analysis..."
        self.text_input.insert('1.0', placeholder_text)
        self.text_input.bind('<FocusIn>', self.clear_placeholder)
        self.text_input.bind('<FocusOut>', self.add_placeholder)
        self.placeholder_active = True

    def clear_placeholder(self, event):
        if self.placeholder_active:
            self.text_input.delete('1.0', tk.END)
            self.text_input.config(fg='#111827')
            self.placeholder_active = False

    def add_placeholder(self, event):
        if not self.text_input.get('1.0', tk.END).strip():
            placeholder_text = "Type or paste your text here for manual analysis..."
            self.text_input.insert('1.0', placeholder_text)
            self.text_input.config(fg='#9ca3af')
            self.placeholder_active = True

    def create_control_buttons(self):
        button_container = tk.Frame(self.main_frame, bg='#f8fafc')
        button_container.pack(fill='x', pady=20)

        button_frame = tk.Frame(button_container, bg='#f8fafc')
        button_frame.pack()

        self.analyze_button = self.create_modern_button(
            button_frame,
            "🔍 ANALYZE TEXT",
            self.analyze_text,
            "#3b82f6",
            "#60a5fa",
            text_color="white",
            width=22,
            height=2
        )
        self.analyze_button.pack(side='left', padx=12)

        clear_button = self.create_modern_button(
            button_frame,
            "🗑️ CLEAR TEXT",
            self.clear_text,
            "#ec4899",
            "#f472b6",
            text_color="white",
            width=17,
            height=2
        )
        clear_button.pack(side='left', padx=12)

        sample_button = self.create_modern_button(
            button_frame,
            "📄 LOAD SAMPLE",
            self.load_sample_text,
            "#10b981",
            "#34d399",
            text_color="white",
            width=20,
            height=2
        )
        sample_button.pack(side='left', padx=12)

    def create_results_section(self):
        results_container = tk.Frame(self.main_frame, bg='#f8fafc')
        results_container.pack(fill='both', expand=True, pady=(20, 0))

        results_frame = tk.Frame(results_container, bg='#fdf4ff', relief='raised', bd=3)
        results_frame.pack(fill='both', expand=True, padx=25)

        results_header = tk.Frame(results_frame, bg='#fdf4ff')
        results_header.pack(fill='x', padx=25, pady=(20, 15))

        results_icon = tk.Label(
            results_header,
            text="🎯",
            font=("Segoe UI", 16),
            bg='#fdf4ff',
            fg='#c026d3'
        )
        results_icon.pack(side='left')

        results_label = tk.Label(
            results_header,
            text="Analysis Results:",
            font=("Segoe UI", 16, "bold"),
            bg='#fdf4ff',
            fg='#86198f'
        )
        results_label.pack(side='left', padx=(12, 0))

        self.result_display_frame = tk.Frame(results_frame, bg='#ffffff', relief='sunken', bd=3, height=120)
        self.result_display_frame.pack(fill='x', padx=25, pady=(0, 20))
        self.result_display_frame.pack_propagate(False)

        self.result_var = tk.StringVar(value="⏳ No analysis performed yet")
        self.result_label = tk.Label(
            self.result_display_frame,
            textvariable=self.result_var,
            font=("Segoe UI", 20, "bold"),
            bg='#ffffff',
            fg='#6b7280',
            wraplength=650,
            justify='center'
        )
        self.result_label.pack(expand=True)

        self.details_var = tk.StringVar()
        self.details_label = tk.Label(
            self.result_display_frame,
            textvariable=self.details_var,
            font=("Segoe UI", 13, "italic"),
            bg='#ffffff',
            fg='#9ca3af'
        )
        self.details_label.pack()

    def create_footer(self):
        footer_frame = tk.Frame(self.main_frame, bg='#e5e7eb', height=75, relief='raised', bd=3)
        footer_frame.pack(fill='x', side='bottom', pady=(20, 0))
        footer_frame.pack_propagate(False)

        status_container = tk.Frame(footer_frame, bg='#e5e7eb')
        status_container.pack(fill='x', padx=25, pady=12)

        self.status_var = tk.StringVar(value="🟢 Ready - Upload file or enter text manually")
        status_label = tk.Label(
            status_container,
            textvariable=self.status_var,
            font=("Segoe UI", 11),
            bg='#e5e7eb',
            fg='#374151',
            anchor='w'
        )
        status_label.pack(side='left', fill='x', expand=True)

        exit_button = self.create_modern_button(
            status_container,
            "❌ EXIT",
            self.exit_application,
            "#ef4444",
            "#f87171",
            text_color="white",
            width=14,
            height=1
        )
        exit_button.pack(side='right')

    def show_result_popup(self, result, text_analyzed, analysis_type="Text"):
        popup = tk.Toplevel(self.root)
        popup.title("🎯 Analysis Result")
        popup.geometry("650x550")
        popup.configure(bg='#f9fafb')
        popup.resizable(False, False)

        popup.transient(self.root)
        popup.grab_set()

        if result == "Bullying":
            header_bg = '#fee2e2'
            content_bg = '#fef2f2'
            accent_color = '#dc2626'
            result_text = "⚠️ BULLYING DETECTED"
            icon = "🚨"
        else:
            header_bg = '#d1fae5'
            content_bg = '#f0fdf4'
            accent_color = '#059669'
            result_text = "✅ CONTENT IS SAFE"
            icon = "🛡️"

        header_frame = tk.Frame(popup, bg=header_bg, height=110)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)

        result_frame = tk.Frame(header_frame, bg=header_bg)
        result_frame.pack(expand=True)

        icon_label = tk.Label(
            result_frame,
            text=icon,
            font=("Segoe UI", 36),
            bg=header_bg,
            fg=accent_color
        )
        icon_label.pack(pady=(12, 8))

        result_label = tk.Label(
            result_frame,
            text=result_text,
            font=("Segoe UI", 20, "bold"),
            bg=header_bg,
            fg=accent_color
        )
        result_label.pack()

        content_frame = tk.Frame(popup, bg=content_bg)
        content_frame.pack(fill='both', expand=True, padx=25, pady=25)

        type_label = tk.Label(
            content_frame,
            text=f"Analysis Type: {analysis_type}",
            font=("Segoe UI", 14, "bold"),
            bg=content_bg,
            fg=accent_color
        )
        type_label.pack(anchor='w', pady=(0, 12))

        text_label = tk.Label(
            content_frame,
            text="Analyzed Content:",
            font=("Segoe UI", 14, "bold"),
            bg=content_bg,
            fg='#374151'
        )
        text_label.pack(anchor='w', pady=(12, 8))

        text_frame = tk.Frame(content_frame, bg=content_bg)
        text_frame.pack(fill='both', expand=True, pady=(0, 18))

        text_display = scrolledtext.ScrolledText(
            text_frame,
            height=8,
            font=("Consolas", 11),
            wrap=tk.WORD,
            bg='#ffffff',
            fg='#374151',
            relief='sunken',
            bd=2,
            padx=12,
            pady=12,
            state='normal'
        )
        text_display.pack(fill='both', expand=True)

        display_text = text_analyzed[:1000] + "..." if len(text_analyzed) > 1000 else text_analyzed
        text_display.insert('1.0', display_text)
        text_display.config(state='disabled')

        if result == "Bullying":
            details_text = "⚠️ This content contains potentially harmful or bullying language. Please review and take appropriate action."
            details_color = '#dc2626'
        else:
            details_text = "✅ This content appears to be safe and does not contain bullying or toxic language."
            details_color = '#059669'

        details_label = tk.Label(
            content_frame,
            text=details_text,
            font=("Segoe UI", 12),
            bg=content_bg,
            fg=details_color,
            wraplength=580,
            justify='left'
        )
        details_label.pack(anchor='w', pady=(0, 18))

        button_frame = tk.Frame(content_frame, bg=content_bg)
        button_frame.pack(fill='x')

        def copy_result():
            popup.clipboard_clear()
            popup.clipboard_append(f"Analysis Result: {result}\nContent: {text_analyzed}")
            messagebox.showinfo("Copied", "Result copied to clipboard!", parent=popup)

        copy_button = self.create_modern_button(
            button_frame,
            "📋 COPY",
            copy_result,
            '#6366f1',
            '#818cf8',
            text_color='white',
            width=14,
            height=2
        )
        copy_button.pack(side='right', padx=(0, 12))

        close_button = self.create_modern_button(
            button_frame,
            "✓ CLOSE",
            popup.destroy,
            accent_color,
            '#22c55e' if result == "Non-Bullying" else '#f87171',
            text_color='white',
            width=16,
            height=2
        )
        close_button.pack(side='right')

        popup.update_idletasks()
        x = (popup.winfo_screenwidth() // 2) - (popup.winfo_width() // 2)
        y = (popup.winfo_screenheight() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f"+{x}+{y}")

    def upload_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Text File for Bullying Analysis",
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ],
            initialdir=os.getcwd()
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()

                self.current_file = {
                    'path': file_path,
                    'name': os.path.basename(file_path),
                    'content': content,
                    'size': len(content)
                }

                file_info = f"📁 {self.current_file['name']} | Size: {self.current_file['size']:,} characters"
                self.file_info_var.set(file_info)

                self.analyze_file_button.config(state='normal')
                self.status_var.set(f"📁 File loaded: {self.current_file['name']}")

                self.text_input.delete('1.0', tk.END)
                preview = content[:500] + "..." if len(content) > 500 else content
                self.text_input.insert('1.0',
                                       f"FILE PREVIEW: {self.current_file['name']}\n" + "=" * 50 + "\n" + preview)
                self.text_input.config(fg='#111827')
                self.placeholder_active = False

            except Exception as e:
                messagebox.showerror("File Error", f"Could not read file: {str(e)}")
                self.status_var.set("❌ File loading failed")

    def analyze_file(self):
        if not self.current_file:
            messagebox.showwarning("No File", "Please upload a file first!")
            return

        self.status_var.set("🔄 Analyzing file content... Please wait")
        self.analyze_file_button.config(state='disabled', text='🔄 ANALYZING...')
        self.result_var.set("🔄 Processing file content...")
        self.result_label.config(fg='#f59e0b')
        self.root.update()

        try:
            file_content = self.current_file['content']
            overall_result = self.predict_toxicity(file_content)

            lines = [line.strip() for line in file_content.split('\n') if line.strip()]

            bullying_lines = []
            safe_lines = []
            total_lines = len(lines)

            for i, line in enumerate(lines):
                if len(line) > 10:
                    result = self.predict_toxicity(line)
                    if result == "Bullying":
                        bullying_lines.append((i + 1, line[:100] + "..." if len(line) > 100 else line))
                    elif result == "Non-Bullying":
                        safe_lines.append((i + 1, line[:100] + "..." if len(line) > 100 else line))

            bullying_count = len(bullying_lines)
            safe_count = len(safe_lines)
            total_analyzed = bullying_count + safe_count

            if bullying_count > 0:
                self.result_var.set(f"⚠️ BULLYING CONTENT DETECTED")
                self.result_label.config(fg='#dc2626', bg='#fef2f2')
                self.result_display_frame.config(bg='#fef2f2')
                self.details_label.config(bg='#fef2f2')

                percentage = (bullying_count / total_analyzed * 100) if total_analyzed > 0 else 0
                details = f"Found {bullying_count} bullying lines out of {total_analyzed} analyzed ({percentage:.1f}%)"
                self.details_var.set(details)

                self.show_result_popup("Bullying", file_content, f"File Analysis: {self.current_file['name']}")
                self.show_detailed_results(bullying_lines, safe_lines, self.current_file['name'])
                self.status_var.set(f"🔴 File analysis complete - {bullying_count} bullying instances found")

            else:
                self.result_var.set("✅ FILE CONTENT IS SAFE")
                self.result_label.config(fg='#059669', bg='#f0fdf4')
                self.result_display_frame.config(bg='#f0fdf4')
                self.details_label.config(bg='#f0fdf4')
                self.details_var.set(f"Analyzed {total_analyzed} lines - No bullying content detected")

                self.show_result_popup("Non-Bullying", file_content, f"File Analysis: {self.current_file['name']}")
                self.status_var.set("🟢 File analysis complete - Content is safe")

        except Exception as e:
            messagebox.showerror("Analysis Error", f"File analysis failed: {str(e)}")
            self.status_var.set("🔴 File analysis error occurred")

        finally:
            self.analyze_file_button.config(state='normal', text='🔍 ANALYZE FILE')

    def show_detailed_results(self, bullying_lines, safe_lines, filename):
        results_window = tk.Toplevel(self.root)
        results_window.title(f"📊 Detailed Analysis - {filename}")
        results_window.geometry("800x600")
        results_window.configure(bg='#f9fafb')

        header = tk.Label(
            results_window,
            text=f"📊 Detailed Analysis Results - {filename}",
            font=("Segoe UI", 18, "bold"),
            bg='#e0f2fe',
            fg='#0369a1',
            pady=18
        )
        header.pack(fill='x')

        notebook = ttk.Notebook(results_window)
        notebook.pack(fill='both', expand=True, padx=25, pady=25)

        if bullying_lines:
            bullying_frame = tk.Frame(notebook, bg='#fef2f2')
            notebook.add(bullying_frame, text=f"⚠️ Bullying Lines ({len(bullying_lines)})")

            bullying_text = scrolledtext.ScrolledText(
                bullying_frame,
                font=("Consolas", 11),
                bg='#ffffff',
                fg='#dc2626',
                wrap=tk.WORD
            )
            bullying_text.pack(fill='both', expand=True, padx=12, pady=12)

            bullying_text.insert('1.0',
                                 f"BULLYING CONTENT DETECTED IN {len(bullying_lines)} LINES:\n" + "=" * 60 + "\n\n")
            for line_num, content in bullying_lines:
                bullying_text.insert(tk.END, f"Line {line_num}: {content}\n\n")

        if safe_lines:
            safe_frame = tk.Frame(notebook, bg='#f0fdf4')
            notebook.add(safe_frame, text=f"✅ Safe Lines ({len(safe_lines)})")

            safe_text = scrolledtext.ScrolledText(
                safe_frame,
                font=("Consolas", 11),
                bg='#ffffff',
                fg='#059669',
                wrap=tk.WORD
            )
            safe_text.pack(fill='both', expand=True, padx=12, pady=12)

            safe_text.insert('1.0', f"SAFE CONTENT IN {len(safe_lines)} LINES:\n" + "=" * 50 + "\n\n")
            for line_num, content in safe_lines[:50]:
                safe_text.insert(tk.END, f"Line {line_num}: {content}\n\n")

            if len(safe_lines) > 50:
                safe_text.insert(tk.END, f"\n... and {len(safe_lines) - 50} more safe lines")

    def clear_file(self):
        self.current_file = None
        self.file_info_var.set("No file selected")
        self.analyze_file_button.config(state='disabled')
        self.status_var.set("🟢 File cleared - Ready for new upload")

    def load_sample_text(self):
        sample_texts = [
            "You are such an idiot! I hate you!",
            "Have a wonderful day! You're amazing!",
            "Stop bothering me, you loser!",
            "Thanks for your help, I really appreciate it.",
            "You're so stupid, nobody likes you!"
        ]

        if self.placeholder_active:
            self.text_input.delete('1.0', tk.END)
            self.placeholder_active = False

        import random
        sample = random.choice(sample_texts)
        self.text_input.delete('1.0', tk.END)
        self.text_input.insert('1.0', sample)
        self.text_input.config(fg='#111827')

        self.status_var.set(f"📄 Sample text loaded: '{sample[:30]}...'")

    def analyze_text(self):
        text = self.text_input.get('1.0', tk.END).strip()

        if self.placeholder_active or not text:
            messagebox.showwarning("⚠️ Warning", "Please enter some text to analyze!")
            return

        self.status_var.set("🔄 Analyzing text... Please wait")
        self.analyze_button.config(state='disabled', text='🔄 ANALYZING...')
        self.result_var.set("🔄 Processing...")
        self.result_label.config(fg='#f59e0b')
        self.root.update()

        try:
            result = self.predict_toxicity(text)

            if result == "Bullying":
                self.result_var.set("⚠️ BULLYING DETECTED")
                self.result_label.config(fg='#dc2626', bg='#fef2f2')
                self.result_display_frame.config(bg='#fef2f2')
                self.details_label.config(bg='#fef2f2')
                self.details_var.set("⚠️ This content contains toxic/bullying language")
                self.status_var.set("🔴 Analysis complete - Bullying content detected")

            elif result == "Non-Bullying":
                self.result_var.set("✅ CONTENT IS SAFE")
                self.result_label.config(fg='#059669', bg='#f0fdf4')
                self.result_display_frame.config(bg='#f0fdf4')
                self.details_label.config(bg='#f0fdf4')
                self.details_var.set("✅ This content appears to be non-toxic")
                self.status_var.set("🟢 Analysis complete - Safe content detected")

            else:
                self.result_var.set("❓ ANALYSIS ERROR")
                self.result_label.config(fg='#f59e0b', bg='#fffbeb')
                self.result_display_frame.config(bg='#fffbeb')
                self.details_label.config(bg='#fffbeb')
                self.details_var.set(f"Error: {result}")
                self.status_var.set("🟡 Analysis error occurred")

            if result in ["Bullying", "Non-Bullying"]:
                self.show_result_popup(result, text, "Manual Text Input")

        except Exception as e:
            messagebox.showerror("❌ Error", f"Analysis failed: {str(e)}")
            self.status_var.set("🔴 Error occurred during analysis")

        finally:
            self.analyze_button.config(state='normal', text='🔍 ANALYZE TEXT')

    def clear_text(self):
        self.text_input.delete('1.0', tk.END)
        self.add_placeholder(None)

        self.result_var.set("⏳ No analysis performed yet")
        self.result_label.config(fg='#6b7280', bg='#ffffff')
        self.result_display_frame.config(bg='#ffffff')
        self.details_label.config(bg='#ffffff')
        self.details_var.set("")
        self.status_var.set("🟢 Ready - Upload file or enter text manually")

    def exit_application(self):
        if messagebox.askyesno("🚪 Exit Confirmation", "Are you sure you want to exit the Toxicity Detection System?"):
            self.status_var.set("👋 Goodbye! Closing application...")
            self.root.update()
            self.root.after(1000, self.root.destroy)

    def run(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

        self.root.mainloop()


if __name__ == "__main__":
    app = ToxicityGUI()
    app.run()
