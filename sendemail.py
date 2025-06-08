import os
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog, StringVar, OptionMenu
from tkinter import ttk
import threading
import time

class AttendanceEmailNotifier:
    """
    System to send email notifications when students attend class
    Integrates with the existing EmailManager system
    """
    
    def __init__(self, email_file="student_emails.csv", attendance_file="attendance_log.csv"):
        """
        Initialize the attendance email notifier
        
        Parameters:
        - email_file: Path to the CSV containing student email addresses
        - attendance_file: Path to the attendance log CSV
        """
        self.email_file = email_file
        self.attendance_file = attendance_file
        self.student_emails = {}
        self.smtp_settings = {
            "server": "",
            "port": 587,
            "username": "",
            "password": "",
            "sender_email": "",
            "use_tls": True
        }
        self.load_student_emails()
    
    def load_student_emails(self):
        """
        Load student emails from the email file
        
        Returns:
        - Dictionary with student email information
        """
        self.student_emails = {}
        
        # Check if email file exists and load it
        if os.path.exists(self.email_file):
            try:
                df = pd.read_csv(self.email_file)
                for _, row in df.iterrows():
                    if 'Name' in row and 'Email' in row:
                        name = row['Name']
                        email = row['Email']
                        if name and email:  # Only add if both name and email are present
                            self.student_emails[name] = email
                print(f"Loaded {len(self.student_emails)} student emails from {self.email_file}")
            except Exception as e:
                print(f"Error loading student emails: {str(e)}")
                return {}
        else:
            print(f"Email file {self.email_file} not found.")
            return {}
            
        return self.student_emails
    
    def load_attendance_records(self, date=None):
        """
        Load attendance records for the given date or the latest date
        
        Parameters:
        - date: Date string in format YYYY-MM-DD or None for latest date
        
        Returns:
        - DataFrame with attendance records for the specified date
        """
        if not os.path.exists(self.attendance_file):
            print(f"Attendance file {self.attendance_file} not found")
            return pd.DataFrame()
            
        try:
            # Load attendance records
            df = pd.read_csv(self.attendance_file)
            
            # Check if the Date column exists
            if 'Date' not in df.columns:
                print("No Date column found in attendance records")
                return pd.DataFrame()
                
            # If no date provided, get the latest date
            if date is None:
                date = df['Date'].max()
                
            # Filter by date
            daily_records = df[df['Date'] == date].copy()
            print(f"Found {len(daily_records)} attendance records for {date}")
            
            return daily_records
            
        except Exception as e:
            print(f"Error reading attendance records: {str(e)}")
            return pd.DataFrame()
    
    def get_available_dates(self):
        """
        Get a list of available dates in the attendance records
        
        Returns:
        - List of date strings
        """
        if not os.path.exists(self.attendance_file):
            return []
            
        try:
            df = pd.read_csv(self.attendance_file)
            if 'Date' not in df.columns:
                return []
                
            dates = sorted(df['Date'].unique(), reverse=True)
            return dates
        except Exception as e:
            print(f"Error getting available dates: {str(e)}")
            return []
    
    def test_smtp_connection(self):
        """
        Test the SMTP connection with current settings
        
        Returns:
        - Tuple: (success, message)
        """
        try:
            # Validate settings
            if not all([
                self.smtp_settings['server'], 
                self.smtp_settings['port'],
                self.smtp_settings['username'],
                self.smtp_settings['password'],
                self.smtp_settings['sender_email']
            ]):
                return False, "SMTP settings are incomplete"
                
            # Try to connect
            if self.smtp_settings['use_tls']:
                server = smtplib.SMTP(
                    self.smtp_settings['server'], 
                    self.smtp_settings['port']
                )
                server.starttls()
            else:
                server = smtplib.SMTP(
                    self.smtp_settings['server'], 
                    self.smtp_settings['port']
                )
                
            server.login(
                self.smtp_settings['username'],
                self.smtp_settings['password']
            )
            
            server.quit()
            return True, "SMTP connection successful"
            
        except Exception as e:
            return False, f"SMTP connection failed: {str(e)}"
    
    def send_attendance_email(self, student_name, attendance_data):
        """
        Send an attendance notification email to a student
        
        Parameters:
        - student_name: Name of the student
        - attendance_data: Dictionary with attendance details
        
        Returns:
        - Boolean: True if sent successfully, False otherwise
        """
        if student_name not in self.student_emails:
            print(f"No email address found for {student_name}")
            return False
            
        recipient_email = self.student_emails[student_name]
        
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = self.smtp_settings['sender_email']
        msg['To'] = recipient_email
        msg['Subject'] = f"Attendance Confirmation - {attendance_data['Date']}"
        
        # Build email body
        email_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6;">
            <h2>Attendance Confirmation</h2>
            <p>Hello {student_name},</p>
            <p>This email confirms your attendance in class:</p>
            
            <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 15px 0;">
                <p><strong>Date:</strong> {attendance_data['Date']}</p>
                <p><strong>Time:</strong> {attendance_data['Time']}</p>
                <p><strong>Status:</strong> <span style="color: green; font-weight: bold;">Present</span></p>
            </div>
            
            <p>Thank you for your attendance and participation.</p>
            <p>This is an automated message, please do not reply.</p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(email_body, 'html'))
        
        try:
            # Connect to SMTP server
            if self.smtp_settings['use_tls']:
                server = smtplib.SMTP(
                    self.smtp_settings['server'], 
                    self.smtp_settings['port']
                )
                server.starttls()
            else:
                server = smtplib.SMTP(
                    self.smtp_settings['server'], 
                    self.smtp_settings['port']
                )
                
            server.login(
                self.smtp_settings['username'],
                self.smtp_settings['password']
            )
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            print(f"Successfully sent attendance email to {student_name} at {recipient_email}")
            return True
            
        except Exception as e:
            print(f"Failed to send email to {student_name}: {str(e)}")
            return False
    
    def send_batch_emails(self, date, students=None, progress_callback=None):
        """
        Send attendance emails to multiple students
        
        Parameters:
        - date: Date string for attendance records
        - students: List of student names or None for all students with attendance
        - progress_callback: Function to call with progress updates
        
        Returns:
        - Dictionary with results
        """
        # Load attendance records for the specified date
        attendance_records = self.load_attendance_records(date)
        
        if attendance_records.empty:
            return {"success": 0, "failed": 0, "not_found": 0, "message": "No attendance records found"}
            
        # Make sure we have the student emails
        self.load_student_emails()
        
        if not self.student_emails:
            return {"success": 0, "failed": 0, "not_found": 0, "message": "No student emails found"}
            
        # Filter students if a list is provided
        if students:
            attendance_records = attendance_records[attendance_records['Name'].isin(students)]
            
        # Track results
        results = {
            "success": 0,
            "failed": 0,
            "not_found": 0,
            "students": []
        }
        
        total_students = len(attendance_records)
        
        # Send emails for each student with attendance
        for i, (_, record) in enumerate(attendance_records.iterrows()):
            student_name = record['Name']
            
            # Skip if no email
            if student_name not in self.student_emails:
                results["not_found"] += 1
                results["students"].append({
                    "name": student_name,
                    "status": "no_email",
                    "message": "No email address found"
                })
                continue
                
            # Convert record to dict for email
            attendance_data = record.to_dict()
            
            # Report progress
            if progress_callback:
                progress = (i + 1) / total_students * 100
                progress_callback(progress, f"Sending email to {student_name}...")
                
            # Send email
            success = self.send_attendance_email(student_name, attendance_data)
            
            # Track result
            if success:
                results["success"] += 1
                results["students"].append({
                    "name": student_name,
                    "status": "sent",
                    "message": "Email sent successfully"
                })
            else:
                results["failed"] += 1
                results["students"].append({
                    "name": student_name,
                    "status": "failed",
                    "message": "Failed to send email"
                })
                
            # Small delay to avoid overwhelming the server
            time.sleep(0.1)
        
        # Final progress update
        if progress_callback:
            progress_callback(100, "Completed sending emails")
            
        return results
    
    def show_smtp_settings_dialog(self, parent=None):
        """
        Display a dialog to configure SMTP settings
        
        Parameters:
        - parent: Parent tkinter window or None
        
        Returns:
        - Boolean: True if settings were saved, False otherwise
        """
        # Create dialog window
        dialog = tk.Toplevel(parent)
        dialog.title("SMTP Server Settings")
        dialog.geometry("450x400")
        dialog.resizable(False, False)
        dialog.transient(parent)
        dialog.grab_set()
        
        # Make dialog modal
        dialog.focus_set()
        
        # Main frame
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a bold title
        title_label = ttk.Label(
            main_frame, 
            text="Email Server Configuration",
            font=("Arial", 12, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))
        
        # Server settings
        ttk.Label(main_frame, text="SMTP Server:").grid(row=1, column=0, sticky=tk.W, pady=5)
        server_var = tk.StringVar(value=self.smtp_settings["server"])
        server_entry = ttk.Entry(main_frame, textvariable=server_var, width=30)
        server_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Port:").grid(row=2, column=0, sticky=tk.W, pady=5)
        port_var = tk.StringVar(value=str(self.smtp_settings["port"]))
        port_entry = ttk.Entry(main_frame, textvariable=port_var, width=10)
        port_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # TLS option
        use_tls_var = tk.BooleanVar(value=self.smtp_settings["use_tls"])
        tls_check = ttk.Checkbutton(
            main_frame, 
            text="Use TLS encryption", 
            variable=use_tls_var
        )
        tls_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Authentication settings
        auth_frame = ttk.LabelFrame(main_frame, text="Authentication", padding=10)
        auth_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        ttk.Label(auth_frame, text="Username:").grid(row=0, column=0, sticky=tk.W, pady=5)
        username_var = tk.StringVar(value=self.smtp_settings["username"])
        username_entry = ttk.Entry(auth_frame, textvariable=username_var, width=30)
        username_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(auth_frame, text="Password:").grid(row=1, column=0, sticky=tk.W, pady=5)
        password_var = tk.StringVar(value=self.smtp_settings["password"])
        password_entry = ttk.Entry(auth_frame, textvariable=password_var, width=30, show="•")
        password_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Sender settings
        sender_frame = ttk.LabelFrame(main_frame, text="Sender Information", padding=10)
        sender_frame.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        ttk.Label(sender_frame, text="Sender Email:").grid(row=0, column=0, sticky=tk.W, pady=5)
        sender_var = tk.StringVar(value=self.smtp_settings["sender_email"])
        sender_entry = ttk.Entry(sender_frame, textvariable=sender_var, width=30)
        sender_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Status message
        status_var = tk.StringVar()
        status_label = ttk.Label(main_frame, textvariable=status_var, foreground="blue", wraplength=400)
        status_label.grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        # Test connection
        def test_connection():
            # Update settings with current values
            self.smtp_settings = {
                "server": server_var.get(),
                "port": int(port_var.get() or 587),
                "username": username_var.get(),
                "password": password_var.get(),
                "sender_email": sender_var.get(),
                "use_tls": use_tls_var.get()
            }
            
            # Test connection
            status_var.set("Testing connection...")
            dialog.update_idletasks()
            
            # Run test in a separate thread to avoid freezing UI
            def run_test():
                success, message = self.test_smtp_connection()
                
                if success:
                    status_var.set(message)
                    status_label.configure(foreground="green")
                else:
                    status_var.set(message)
                    status_label.configure(foreground="red")
            
            threading.Thread(target=run_test).start()
        
        # Save settings
        def save_settings():
            try:
                # Validate port number
                port = int(port_var.get() or 587)
                
                # Update settings
                self.smtp_settings = {
                    "server": server_var.get(),
                    "port": port,
                    "username": username_var.get(),
                    "password": password_var.get(),
                    "sender_email": sender_var.get(),
                    "use_tls": use_tls_var.get()
                }
                
                dialog.destroy()
                
            except ValueError:
                status_var.set("Invalid port number")
                status_label.configure(foreground="red")
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=2, sticky=tk.E, pady=10)
        
        test_button = ttk.Button(button_frame, text="Test Connection", command=test_connection)
        test_button.pack(side=tk.LEFT, padx=5)
        
        cancel_button = ttk.Button(button_frame, text="Cancel", command=dialog.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5)
        
        save_button = ttk.Button(button_frame, text="Save", command=save_settings)
        save_button.pack(side=tk.LEFT, padx=5)
        
        # Center dialog on parent
        dialog.update_idletasks()
        if parent:
            x = parent.winfo_x() + (parent.winfo_width() // 2) - (dialog.winfo_width() // 2)
            y = parent.winfo_y() + (parent.winfo_height() // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
        
        # Give focus to server entry
        server_entry.focus_set()
        
        # Wait for dialog to be closed
        dialog.wait_window()
        
        # Return True if settings were saved
        return True
    
    def show_email_notifier_dialog(self):
        """
        Display the main dialog for sending attendance emails
        """
        # Create root window
        root = tk.Tk()
        root.title("Attendance Email Notifier")
        root.geometry("800x600")
        root.minsize(700, 500)
        
        # Configure style
        style = ttk.Style()
        style.configure("TButton", padding=5)
        style.configure("TFrame", padding=5)
        
        # Main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header frame
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Create a bold title
        title_label = ttk.Label(
            header_frame, 
            text="Student Attendance Email Notifier",
            font=("Arial", 16, "bold")
        )
        title_label.pack(side=tk.LEFT)
        
        # Settings button
        settings_button = ttk.Button(
            header_frame, 
            text="SMTP Settings",
            command=lambda: self.show_smtp_settings_dialog(root)
        )
        settings_button.pack(side=tk.RIGHT)
        
        # Date selection frame
        date_frame = ttk.LabelFrame(main_frame, text="Select Attendance Date", padding=10)
        date_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Get available dates
        dates = self.get_available_dates()
        
        if not dates:
            date_label = ttk.Label(
                date_frame, 
                text="No attendance records found. Please record attendance first.",
                foreground="red"
            )
            date_label.pack(pady=10)
        else:
            # Create date selector
            date_var = tk.StringVar(value=dates[0] if dates else "")
            date_dropdown = ttk.Combobox(date_frame, textvariable=date_var, values=dates, width=15)
            date_dropdown.pack(side=tk.LEFT, padx=5)
            
            # Refresh button
            refresh_button = ttk.Button(
                date_frame, 
                text="Refresh",
                command=lambda: date_dropdown.configure(values=self.get_available_dates())
            )
            refresh_button.pack(side=tk.LEFT, padx=5)
            
            # Load button
            load_button = ttk.Button(
                date_frame, 
                text="Load Students",
                command=lambda: load_attendance_records()
            )
            load_button.pack(side=tk.LEFT, padx=5)
        
        # Students frame
        students_frame = ttk.LabelFrame(main_frame, text="Students", padding=10)
        students_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Create treeview for students
        columns = ("Name", "Status", "Email")
        student_tree = ttk.Treeview(students_frame, columns=columns, show="headings")
        
        # Define headings
        for col in columns:
            student_tree.heading(col, text=col)
            
        # Set column widths
        student_tree.column("Name", width=200)
        student_tree.column("Status", width=100)
        student_tree.column("Email", width=250)
        
        # Add scrollbars
        y_scrollbar = ttk.Scrollbar(students_frame, orient=tk.VERTICAL, command=student_tree.yview)
        student_tree.configure(yscrollcommand=y_scrollbar.set)
        
        # Position treeview and scrollbar
        student_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Status message
        status_var = tk.StringVar()
        status_label = ttk.Label(status_frame, textvariable=status_var)
        status_label.pack(side=tk.LEFT)
        
        # Progress bar
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            status_frame, 
            orient=tk.HORIZONTAL, 
            length=400, 
            mode='determinate',
            variable=progress_var
        )
        progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        # Function to load attendance records
        def load_attendance_records():
            # Clear existing items
            for item in student_tree.get_children():
                student_tree.delete(item)
                
            # Get selected date
            selected_date = date_var.get()
            
            if not selected_date:
                status_var.set("No date selected")
                return
                
            # Load attendance records
            attendance_records = self.load_attendance_records(selected_date)
            
            if attendance_records.empty:
                status_var.set(f"No attendance records found for {selected_date}")
                return
                
            # Make sure we have the student emails
            self.load_student_emails()
            
            # Add students to treeview
            for _, record in attendance_records.iterrows():
                student_name = record['Name']
                
                # Check if student has email
                email = self.student_emails.get(student_name, "")
                email_status = "✓" if email else "✗"
                
                # Add to treeview
                student_tree.insert("", tk.END, values=(student_name, email_status, email))
                
            status_var.set(f"Loaded {len(attendance_records)} students for {selected_date}")
        
        # Function to update progress
        def update_progress(progress, message):
            progress_var.set(progress)
            status_var.set(message)
            root.update_idletasks()
        
        # Function to send emails
        def send_emails():
            # Validate SMTP settings
            if not all([
                self.smtp_settings['server'], 
                self.smtp_settings['username'],
                self.smtp_settings['password'],
                self.smtp_settings['sender_email']
            ]):
                messagebox.showerror(
                    "SMTP Settings Required", 
                    "Please configure SMTP settings before sending emails."
                )
                return
                
            # Get selected date
            selected_date = date_var.get()
            
            if not selected_date:
                messagebox.showerror("Error", "Please select a date")
                return
                
            # Get selected students or all students
            selected_items = student_tree.selection()
            
            if selected_items:
                # Get names of selected students
                selected_students = []
                for item in selected_items:
                    values = student_tree.item(item, "values")
                    selected_students.append(values[0])  # Name is the first column
                    
                # Confirm with user
                confirm = messagebox.askyesno(
                    "Confirm", 
                    f"Send attendance emails to {len(selected_students)} selected students?"
                )
            else:
                # Confirm sending to all students
                confirm = messagebox.askyesno(
                    "Confirm", 
                    "Send attendance emails to all students with attendance on this date?"
                )
                selected_students = None
                
            if not confirm:
                return
                
            # Disable UI elements during sending
            send_button.config(state=tk.DISABLED)
            select_all_button.config(state=tk.DISABLED)
            
            # Reset progress bar
            progress_var.set(0)
            
            # Send emails in a separate thread to avoid freezing UI
            def send_email_thread():
                # Send emails
                results = self.send_batch_emails(selected_date, selected_students, update_progress)
                
                # Update UI with results
                def update_ui():
                    # Re-enable buttons
                    send_button.config(state=tk.NORMAL)
                    select_all_button.config(state=tk.NORMAL)
                    
                    # Show results
                    success = results["success"]
                    failed = results["failed"]
                    not_found = results["not_found"]
                    
                    # Update status
                    status_var.set(f"Sent: {success}, Failed: {failed}, No Email: {not_found}")
                    
                    # Show message box with summary
                    if success > 0:
                        messagebox.showinfo(
                            "Email Results", 
                            f"Successfully sent {success} emails.\n"
                            f"Failed: {failed}\n"
                            f"No email found: {not_found}"
                        )
                    else:
                        messagebox.showerror(
                            "Email Failed", 
                            f"Failed to send any emails.\n"
                            f"Failed: {failed}\n"
                            f"No email found: {not_found}"
                        )
                    
                    # Refresh student list to show updated status
                    load_attendance_records()
                
                # Schedule UI update on main thread
                root.after(0, update_ui)
            
            # Start the thread
            threading.Thread(target=send_email_thread).start()
        
        # Function to select all students
        def select_all():
            # Select all items
            for item in student_tree.get_children():
                student_tree.selection_add(item)
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X)
        
        # Add buttons
        select_all_button = ttk.Button(buttons_frame, text="Select All", command=select_all)
        select_all_button.pack(side=tk.LEFT, padx=5)
        
        refresh_button = ttk.Button(
            buttons_frame, 
            text="Refresh List", 
            command=lambda: load_attendance_records()
        )
        refresh_button.pack(side=tk.LEFT, padx=5)
        
        exit_button = ttk.Button(buttons_frame, text="Exit", command=root.destroy)
        exit_button.pack(side=tk.RIGHT, padx=5)
        
        send_button = ttk.Button(buttons_frame, text="Send Emails", command=send_emails)
        send_button.pack(side=tk.RIGHT, padx=5)
        
        # Center on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        # If we have dates, load the first one
        if dates:
            load_attendance_records()
        
        # Start the mainloop
        root.mainloop()

# Usage example
if __name__ == "__main__":
    notifier = AttendanceEmailNotifier()
    notifier.show_email_notifier_dialog()