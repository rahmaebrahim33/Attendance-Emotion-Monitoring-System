import smtplib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import pyttsx3
import time
from tkinter import Tk, simpledialog, messagebox, StringVar, Label, Entry, Button, Frame, Listbox, Scrollbar
from tkinter import LEFT, RIGHT, TOP, BOTTOM, X, Y, BOTH, END, SINGLE, W, E, VERTICAL, HORIZONTAL

class BonusFeatures:
    """
    Bonus features for Vision-Based Attendance and Emotion Monitoring System including:
    - Email notifications to absent students
    - Emotion analytics with visualizations
    - Voice feedback for attendance events
    """
    
    def __init__(self, attendance_file="attendance_log.csv"):
        """
        Initialize bonus features module
        
        Parameters:
        - attendance_file: CSV file containing attendance records
        """
        self.attendance_file = attendance_file
        self.email_config = {
            'server': 'smtp.gmail.com',
            'port': 587,
            'username': '',
            'password': '',
            'sender': ''
        }
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        
        # Analytics directory
        self.analytics_dir = "analytics"
        if not os.path.exists(self.analytics_dir):
            os.makedirs(self.analytics_dir)
            
        # Load email configuration if exists
        self._load_email_config()
    
    def _load_email_config(self):
        """Load email configuration from file if exists"""
        try:
            if os.path.exists('email_config.txt'):
                with open('email_config.txt', 'r') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            if key in self.email_config:
                                if key == 'port':
                                    self.email_config[key] = int(value)
                                else:
                                    self.email_config[key] = value
        except Exception as e:
            print(f"Error loading email config: {str(e)}")
    
    def configure_email(self):
        """Set up email configuration through a GUI dialog"""
        root = Tk()
        root.title("Email Configuration")
        root.geometry("400x350")
        root.resizable(False, False)
        
        # Create a frame for the form with some padding
        form_frame = Frame(root, padx=20, pady=20)
        form_frame.pack(fill=X)
        
        # Email server
        Label(form_frame, text="SMTP Server:", anchor=W).grid(row=0, column=0, sticky=W, pady=5)
        server_var = StringVar(value=self.email_config['server'])
        Entry(form_frame, textvariable=server_var, width=30).grid(row=0, column=1, pady=5, padx=5)
        
        # Port
        Label(form_frame, text="Port:", anchor=W).grid(row=1, column=0, sticky=W, pady=5)
        port_var = StringVar(value=str(self.email_config['port']))
        Entry(form_frame, textvariable=port_var, width=30).grid(row=1, column=1, pady=5, padx=5)
        
        # Username (email)
        Label(form_frame, text="Email Username:", anchor=W).grid(row=2, column=0, sticky=W, pady=5)
        username_var = StringVar(value=self.email_config['username'])
        Entry(form_frame, textvariable=username_var, width=30).grid(row=2, column=1, pady=5, padx=5)
        
        # Password
        Label(form_frame, text="Password:", anchor=W).grid(row=3, column=0, sticky=W, pady=5)
        password_var = StringVar(value=self.email_config['password'])
        Entry(form_frame, textvariable=password_var, width=30, show="*").grid(row=3, column=1, pady=5, padx=5)
        
        # Sender address
        Label(form_frame, text="Sender Address:", anchor=W).grid(row=4, column=0, sticky=W, pady=5)
        sender_var = StringVar(value=self.email_config['sender'])
        Entry(form_frame, textvariable=sender_var, width=30).grid(row=4, column=1, pady=5, padx=5)
        
        # Notes frame
        notes_frame = Frame(root, padx=20)
        notes_frame.pack(fill=X)
        
        # Note about app passwords for Gmail
        note_text = "Note: For Gmail, use App Password instead of regular password"
        Label(notes_frame, text=note_text, fg="red", wraplength=350, justify=LEFT).pack(anchor=W, pady=5)
        
        # Buttons frame
        btn_frame = Frame(root, padx=20, pady=10)
        btn_frame.pack(fill=X)
        
        # Save function
        def save_config():
            # Validate input
            if not server_var.get().strip():
                messagebox.showerror("Error", "SMTP Server cannot be empty")
                return
                
            if not username_var.get().strip():
                messagebox.showerror("Error", "Username cannot be empty")
                return
                
            if not password_var.get().strip():
                messagebox.showerror("Error", "Password cannot be empty")
                return
                
            if not sender_var.get().strip():
                messagebox.showerror("Error", "Sender address cannot be empty")
                return
                
            try:
                port = int(port_var.get())
                if port <= 0:
                    raise ValueError("Port must be positive")
            except ValueError:
                messagebox.showerror("Error", "Port must be a valid number")
                return
                
            # Update config
            self.email_config['server'] = server_var.get().strip()
            self.email_config['port'] = port
            self.email_config['username'] = username_var.get().strip()
            self.email_config['password'] = password_var.get()  # Don't strip password
            self.email_config['sender'] = sender_var.get().strip()
            
            # Save to file for future use
            try:
                with open('email_config.txt', 'w') as f:
                    for key, value in self.email_config.items():
                        if key != 'password':  # Don't store password in plain text
                            f.write(f"{key}={value}\n")
                messagebox.showinfo("Success", "Email configuration saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
                
            root.destroy()
            
        # Test connection function
        def test_email():
            if not server_var.get().strip() or not username_var.get().strip() or not password_var.get():
                messagebox.showerror("Error", "Please fill in all required fields")
                return
                
            try:
                # Temporarily update config with form values
                temp_config = {
                    'server': server_var.get().strip(),
                    'port': int(port_var.get()),
                    'username': username_var.get().strip(),
                    'password': password_var.get(),
                    'sender': sender_var.get().strip() or username_var.get().strip()
                }
                
                # Create SMTP connection
                server = smtplib.SMTP(temp_config['server'], temp_config['port'])
                server.ehlo()
                server.starttls()
                server.login(temp_config['username'], temp_config['password'])
                server.quit()
                
                messagebox.showinfo("Success", "Email configuration test successful")
            except Exception as e:
                messagebox.showerror("Test Failed", f"Connection test failed: {str(e)}")
        
        # Add buttons
        Button(btn_frame, text="Test Connection", command=test_email, width=15).pack(side=LEFT, padx=5)
        Button(btn_frame, text="Save", command=save_config, width=15).pack(side=RIGHT, padx=5)
        Button(btn_frame, text="Cancel", command=root.destroy, width=15).pack(side=RIGHT, padx=5)
        
        # Make it modal and center on screen
        root.transient()
        root.grab_set()
        root.focus_set()
        
        # Center window
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Run the dialog
        root.mainloop()
    
    def is_valid_email(self, email):
        """Check if email is valid"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email))
        
    def generate_student_emails_dialog(self):
        """Open dialog to collect student email addresses with improved interface"""
        # Load existing data
        students = {}
        email_file = 'student_emails.csv'
        
        if os.path.exists(self.attendance_file):
            df = pd.read_csv(self.attendance_file)
            unique_names = sorted(df['Name'].unique())
            
            # Try to load existing email data
            if os.path.exists(email_file):
                email_df = pd.read_csv(email_file)
                for _, row in email_df.iterrows():
                    students[row['Name']] = {'email': row['Email']}
        else:
            messagebox.showerror("Error", "Attendance file not found.")
            return
            
        # Create UI for email input
        root = Tk()
        root.title("Student Email Manager")
        root.geometry("600x500")
        
        # Header
        header_frame = Frame(root, pady=10, padx=20)
        header_frame.pack(fill=X)
        Label(header_frame, text="Manage Student Email Addresses", font=("Arial", 14, "bold")).pack(anchor=W)
        Label(header_frame, text="Add, update or remove email addresses for attendance notifications", 
              font=("Arial", 10)).pack(anchor=W, pady=(0, 10))
        
        # Search and filter
        search_frame = Frame(root, pady=5, padx=20)
        search_frame.pack(fill=X)
        
        Label(search_frame, text="Search:").pack(side=LEFT)
        search_var = StringVar()
        
        # Filter student list as user types
        def filter_list(*args):
            search_term = search_var.get().lower()
            student_listbox.delete(0, END)
            for name in unique_names:
                if search_term in name.lower():
                    student_listbox.insert(END, name)
                    
        search_var.trace("w", filter_list)
        Entry(search_frame, textvariable=search_var, width=30).pack(side=LEFT, padx=5)
        
        # Main content frame with two columns
        content_frame = Frame(root, padx=20, pady=10)
        content_frame.pack(fill=BOTH, expand=True)
        
        # Left side: Student list
        list_frame = Frame(content_frame, width=200)
        list_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))
        
        Label(list_frame, text="Students:").pack(anchor=W)
        
        # Create listbox with scrollbar
        list_container = Frame(list_frame)
        list_container.pack(fill=BOTH, expand=True)
        
        scrollbar = Scrollbar(list_container)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        student_listbox = Listbox(list_container, selectmode=SINGLE, yscrollcommand=scrollbar.set)
        student_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=student_listbox.yview)
        
        # Populate listbox
        for name in unique_names:
            student_listbox.insert(END, name)
            
        # Right side: Edit details
        edit_frame = Frame(content_frame, width=300)
        edit_frame.pack(side=RIGHT, fill=BOTH, expand=True)
        
        Label(edit_frame, text="Student Details:").pack(anchor=W)
        
        details_frame = Frame(edit_frame, pady=10)
        details_frame.pack(fill=X)
        
        # Details fields
        Label(details_frame, text="Name:").grid(row=0, column=0, sticky=W, pady=5)
        name_var = StringVar()
        name_label = Label(details_frame, textvariable=name_var, font=("Arial", 10, "bold"))
        name_label.grid(row=0, column=1, sticky=W, pady=5)
        
        Label(details_frame, text="Email:").grid(row=1, column=0, sticky=W, pady=5)
        email_var = StringVar()
        email_entry = Entry(details_frame, textvariable=email_var, width=30)
        email_entry.grid(row=1, column=1, sticky=W, pady=5)
        
        # Status message
        status_var = StringVar()
        status_label = Label(edit_frame, textvariable=status_var, fg="blue")
        status_label.pack(anchor=W, pady=5)
        
        # Handle selection
        def on_select(event):
            try:
                index = student_listbox.curselection()[0]
                selected_name = student_listbox.get(index)
                name_var.set(selected_name)
                
                # Set email if exists
                if selected_name in students:
                    email_var.set(students[selected_name].get('email', ''))
                else:
                    email_var.set('')
                    
                status_var.set("")
            except IndexError:
                pass
                
        student_listbox.bind('<<ListboxSelect>>', on_select)
        
        # Update and save functions
        def update_email():
            selected_indices = student_listbox.curselection()
            if not selected_indices:
                status_var.set("No student selected")
                return
                
            name = student_listbox.get(selected_indices[0])
            email = email_var.get().strip()
            
            # Validate email if not empty
            if email and not self.is_valid_email(email):
                status_var.set("Invalid email format")
                return
                
            # Update dictionary
            if name not in students:
                students[name] = {}
                
            students[name]['email'] = email
            status_var.set(f"Updated email for {name}")
        
        def clear_email():
            selected_indices = student_listbox.curselection()
            if not selected_indices:
                return
                
            name = student_listbox.get(selected_indices[0])
            email_var.set("")
            
            if name in students:
                students[name]['email'] = ""
                status_var.set(f"Cleared email for {name}")
        
        def save_all():
            try:
                # Create data structure for saving
                email_data = []
                for name, info in students.items():
                    email = info.get('email', '').strip()
                    if email:  # Only save if email is provided
                        email_data.append({'Name': name, 'Email': email})
                
                # Add any students with emails entered in this session
                for name in unique_names:
                    if name not in students:
                        continue
                        
                # Save to CSV
                pd.DataFrame(email_data).to_csv('student_emails.csv', index=False)
                messagebox.showinfo("Success", f"Saved email addresses for {len(email_data)} students")
                root.destroy()
            except Exception as e:
                status_var.set(f"Error: {str(e)}")
                messagebox.showerror("Error", f"Failed to save: {str(e)}")
        
        # Button frame
        button_frame = Frame(edit_frame, pady=10)
        button_frame.pack(fill=X)
        
        Button(button_frame, text="Update", command=update_email).pack(side=LEFT, padx=5)
        Button(button_frame, text="Clear", command=clear_email).pack(side=LEFT, padx=5)
        
        # Bottom buttons
        bottom_frame = Frame(root, pady=10, padx=20)
        bottom_frame.pack(fill=X)
        
        Button(bottom_frame, text="Save All", command=save_all, width=15).pack(side=RIGHT, padx=5)
        Button(bottom_frame, text="Cancel", command=root.destroy, width=15).pack(side=RIGHT, padx=5)
        
        # Import/Export features
        def import_csv():
            try:
                import_file = simpledialog.askstring(
                    "Import CSV", 
                    "Enter path to CSV file (with columns 'Name' and 'Email'):",
                    parent=root
                )
                
                if not import_file:
                    return
                    
                if not os.path.exists(import_file):
                    messagebox.showerror("Error", f"File not found: {import_file}")
                    return
                    
                import_df = pd.read_csv(import_file)
                
                if 'Name' not in import_df.columns or 'Email' not in import_df.columns:
                    messagebox.showerror("Error", "CSV must contain 'Name' and 'Email' columns")
                    return
                    
                # Update students dictionary
                count = 0
                for _, row in import_df.iterrows():
                    name = row['Name']
                    email = row['Email']
                    if name in unique_names and email and self.is_valid_email(email):
                        if name not in students:
                            students[name] = {}
                        students[name]['email'] = email
                        count += 1
                        
                status_var.set(f"Imported {count} email addresses")
                messagebox.showinfo("Import Complete", f"Imported {count} email addresses")
                
                # Refresh current selection
                on_select(None)
                
            except Exception as e:
                messagebox.showerror("Import Error", f"Failed to import: {str(e)}")
                
        def export_csv():
            try:
                # Create data for export
                email_data = []
                for name, info in students.items():
                    email = info.get('email', '').strip()
                    if email:  # Only export if email is provided
                        email_data.append({'Name': name, 'Email': email})
                
                if not email_data:
                    messagebox.showinfo("Export", "No email addresses to export")
                    return
                    
                # Create timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                export_file = f"student_emails_export_{timestamp}.csv"
                
                # Save to CSV
                pd.DataFrame(email_data).to_csv(export_file, index=False)
                messagebox.showinfo("Export Complete", f"Exported {len(email_data)} email addresses to {export_file}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
        
        # Add import/export buttons
        import_export_frame = Frame(bottom_frame)
        import_export_frame.pack(side=LEFT)
        
        Button(import_export_frame, text="Import CSV", command=import_csv, width=12).pack(side=LEFT, padx=5)
        Button(import_export_frame, text="Export CSV", command=export_csv, width=12).pack(side=LEFT, padx=5)
        
        # Make root modal
        root.transient()
        root.grab_set()
        root.focus_set()
        
        # Center window
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        root.mainloop()
    
    def send_absence_notification(self, student_data):
        """
        Send email notification to absent students
        
        Parameters:
        - student_data: Dictionary with student information
          {name: {email: 'student@example.com', last_attendance: '2023-05-10', date: '2023-05-15'}}
        
        Returns:
        - success_count: Number of emails successfully sent
        """
        # Check if email is configured
        if not self.email_config['username'] or not self.email_config['password']:
            print("Email not configured. Please configure email first.")
            return 0
            
        # Connect to SMTP server
        try:
            server = smtplib.SMTP(self.email_config['server'], self.email_config['port'])
            server.ehlo()
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            
            success_count = 0
            
            for name, info in student_data.items():
                if 'email' not in info or not info['email']:
                    print(f"No email available for {name}")
                    continue
                    
                # Get the absence date (today or specified)
                absence_date = info.get('date', datetime.datetime.now().strftime("%Y-%m-%d"))
                    
                # Create message
                msg = MIMEMultipart()
                msg['From'] = self.email_config['sender']
                msg['To'] = info['email']
                msg['Subject'] = f"Attendance Alert: {absence_date}"
                
                # Message body
                body = f"""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px;">
                        <h2 style="color: #444;">Attendance Notification</h2>
                        <p>Dear <strong>{name}</strong>,</p>
                        <p>This is to inform you that you were marked absent on <strong>{absence_date}</strong>.</p>
                        <p>Your last recorded attendance was on: <strong>{info.get('last_attendance', 'Not available')}</strong></p>
                        <div style="background-color: #f8f8f8; padding: 10px; border-left: 4px solid #ccc; margin: 15px 0;">
                            <p>If you believe this is an error, please contact the administrator.</p>
                        </div>
                        <p>Best regards,<br>
                        Attendance Monitoring System</p>
                    </div>
                </body>
                </html>
                """
                
                msg.attach(MIMEText(body, 'html'))
                
                # Send email
                try:
                    server.send_message(msg)
                    print(f"Absence notification sent to {name} ({info['email']}) for date {absence_date}")
                    success_count += 1
                except Exception as e:
                    print(f"Failed to send email to {name}: {str(e)}")
            
            # Close connection
            server.quit()
            
            return success_count
            
        except Exception as e:
            print(f"Email connection error: {str(e)}")
            return 0
    
    def check_absent_students(self, specific_date=None):
        """
        Check for absent students and send notifications with date selection
        
        Parameters:
        - specific_date: Optional date string (YYYY-MM-DD) to check for absences
        
        Returns:
        - Number of notifications sent
        """
        if not os.path.exists(self.attendance_file):
            messagebox.showerror("Error", "Attendance file not found")
            return 0
        
        # Load attendance data
        df = pd.read_csv(self.attendance_file)
        
        # Get date to check
        if specific_date:
            target_date = specific_date
        else:
            # Get today's date
            target_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Get list of all students
        all_students = set(df['Name'].unique())
        
        # Check if we have attendance data for the target date
        if target_date not in df['Date'].unique():
            messagebox.showerror("Error", f"No attendance records found for {target_date}")
            return 0
        
        # Get students who attended on target date
        date_attendance = set(df[df['Date'] == target_date]['Name'].unique())
        
        # Identify absent students
        absent_students = all_students - date_attendance
        
        if not absent_students:
            messagebox.showinfo("Information", f"No absent students detected for {target_date}")
            return 0
        
        # Load student emails
        email_file = 'student_emails.csv'
        student_data = {}
        
        if os.path.exists(email_file):
            email_df = pd.read_csv(email_file)
            for _, row in email_df.iterrows():
                name = row['Name']
                email = row['Email']
                if name in absent_students and email:
                    # Get last attendance date for this student
                    student_records = df[df['Name'] == name].sort_values('Date', ascending=False)
                    last_date = student_records['Date'].iloc[0] if not student_records.empty else "No record"
                    
                    student_data[name] = {
                        'email': email,
                        'last_attendance': last_date,
                        'date': target_date
                    }
        
        # If we have emails for absent students, ask for confirmation
        if student_data:
            confirm = messagebox.askyesno(
                "Confirm",
                f"Send absence notifications to {len(student_data)} students for {target_date}?"
            )
            
            if confirm:
                return self.send_absence_notification(student_data)
            else:
                return 0
        else:
            messagebox.showinfo("Information", "No email addresses available for absent students")
            return 0
    
    def absence_notification_dialog(self):
        """Dialog to select date for absence notifications"""
        if not os.path.exists(self.attendance_file):
            messagebox.showerror("Error", "Attendance file not found")
            return
            
        # Create dialog
        root = Tk()
        root.title("Send Absence Notifications")
        root.geometry("400x300")
        
        # Load dataframe to get available dates
        try:
            df = pd.read_csv(self.attendance_file)
            available_dates = sorted(df['Date'].unique(), reverse=True)  # Most recent first
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load attendance data: {str(e)}")
            root.destroy()
            return
            
        # Create main content
        main_frame = Frame(root, padx=20, pady=20)
        main_frame.pack(fill=BOTH, expand=True)
        
        # Header
        Label(main_frame, text="Send Absence Notifications", font=("Arial", 12, "bold")).pack(anchor=W)
        Label(main_frame, text="Select a date to send absence notifications").pack(anchor=W, pady=(0, 10))
        
        # Option frame
        option_frame = Frame(main_frame)
        option_frame.pack(fill=X, pady=10)
        
        # Radio buttons for today or specific date
        date_choice = StringVar(value="today")
        
        def update_date_selection():
            if date_choice.get() == "today":
                date_listbox.config(state="disabled")
            else:
                date_listbox.config(state="normal")
                
        today_radio = Button(option_frame, text="Today", 
                          command=lambda: date_choice.set("today") or update_date_selection())
        today_radio.pack(side=LEFT, padx=5)
        
        specific_radio = Button(option_frame, text="Specific Date", 
                             command=lambda: date_choice.set("specific") or update_date_selection())
        specific_radio.pack(side=LEFT, padx=5)
        
        # Date selection
        date_frame = Frame(main_frame)
        date_frame.pack(fill=BOTH, expand=True, pady=10)
        
        Label(date_frame, text="Available dates:").pack(anchor=W)
        
        # Create listbox with scrollbar for dates
        list_container = Frame(date_frame)
        list_container.pack(fill=BOTH, expand=True)
        
        scrollbar = Scrollbar(list_container)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        date_listbox = Listbox(list_container, selectmode=SINGLE, yscrollcommand=scrollbar.set)
        date_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=date_listbox.yview)
        
        # Populate listbox with dates
        for date in available_dates:
            date_listbox.insert(END, date)
            
        # Initially disable the listbox since "Today" is selected
        date_listbox.config(state="disabled")
        
        # Action buttons
        button_frame = Frame(root, padx=20, pady=10)
        button_frame.pack(fill=X)
        
        # Function to send notifications
        def send_notifications():
            if date_choice.get() == "today":
                # Send for today's date
                self.check_absent_students()
            else:
                # Get selected date
                selected_indices = date_listbox.curselection()
                if not selected_indices:
                    messagebox.showerror("Error", "Please select a date")
                    return
                    
                selected_date = date_listbox.get(selected_indices[0])
                self.check_absent_students(specific_date=selected_date)
                
            root.destroy()
        
        Button(button_frame, text="Send Notifications", command=send_notifications, width=15).pack(side=RIGHT, padx=5)
        Button(button_frame, text="Cancel", command=root.destroy, width=15).pack(side=RIGHT, padx=5)
        
        # Make dialog modal
        root.transient()
        root.grab_set()
        root.focus_set()
        
        # Center window
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        root.mainloop()
    
    def generate_emotion_analytics(self, date=None):
        """
        Generate analytics about emotions from attendance data
        
        Parameters:
        - date: Specific date to analyze (YYYY-MM-DD), None for all data
        
        Returns:
        - Path to saved visualizations
        """
        if not os.path.exists(self.attendance_file):
            print("Attendance file not found")
            return None
        
        # Load attendance data
        df = pd.read_csv(self.attendance_file)
        
        if df.empty:
            print("No attendance data available")
            return None
        
        # Filter for specific date if provided
        if date:
            df_filtered = df[df['Date'] == date]
            if df_filtered.empty:
                print(f"No data found for date {date}")
                return None
        else:
            df_filtered = df
        
        # Create timestamp for filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = []
        
        # 1. Emotion Distribution Pie Chart
        plt.figure(figsize=(10, 7))
        emotion_counts = df_filtered['Emotion'].value_counts()
        colors = plt.cm.tab10(np.linspace(0, 1, len(emotion_counts)))
        
        plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', 
                startangle=140, colors=colors)
        plt.axis('equal')
        
        title = f"Emotion Distribution" + (f" for {date}" if date else "")
        plt.title(title, fontsize=16)
        
        # Save figure
        pie_file = os.path.join(self.analytics_dir, f"emotion_pie_{timestamp}.png")
        plt.savefig(pie_file)
        saved_files.append(pie_file)
        
        # 2. Emotion Timeline (if we have multiple dates)
        if len(df_filtered['Date'].unique()) > 1:
            plt.figure(figsize=(12, 7))
            
            # Group by date and emotion
            timeline_data = df_filtered.groupby(['Date', 'Emotion']).size().unstack().fillna(0)
            
            # Plot stacked bar chart
            timeline_data.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='tab10')
            
            plt.title("Emotion Trends Over Time", fontsize=16)
            plt.xlabel("Date")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.legend(title="Emotion")
            plt.tight_layout()
            
            # Save figure
            timeline_file = os.path.join(self.analytics_dir, f"emotion_timeline_{timestamp}.png")
            plt.savefig(timeline_file)
            saved_files.append(timeline_file)
        
        # 3. Emotion by Student
        if len(df_filtered['Name'].unique()) > 1:
            plt.figure(figsize=(12, 8))
            
            # Group by name and emotion
            student_emotion = df_filtered.groupby(['Name', 'Emotion']).size().unstack().fillna(0)
            
            # Convert to percentage
            student_emotion_pct = student_emotion.div(student_emotion.sum(axis=1), axis=0) * 100
            
            # Plot stacked bar chart
            student_emotion_pct.plot(kind='barh', stacked=True, ax=plt.gca(), colormap='tab10')
            
            plt.title("Emotion Distribution by Student", fontsize=16)
            plt.xlabel("Percentage")
            plt.ylabel("Student")
            plt.legend(title="Emotion")
            plt.tight_layout()
            
            # Save figure
            student_file = os.path.join(self.analytics_dir, f"emotion_by_student_{timestamp}.png")
            plt.savefig(student_file)
            saved_files.append(student_file)
        
        # 4. Summary table
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        
        # Create summary text
        summary = [
            f"Emotion Analytics Summary {'for ' + date if date else ''}",
            f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total records: {len(df_filtered)}",
            f"Date range: {df_filtered['Date'].min()} to {df_filtered['Date'].max()}",
            f"Number of students: {len(df_filtered['Name'].unique())}",
            "\nEmotion Distribution:",
        ]
        
        # Add emotion percentages
        for emotion, count in emotion_counts.items():
            percentage = 100 * count / len(df_filtered)
            summary.append(f"  - {emotion}: {count} records ({percentage:.1f}%)")
            
        # Add most common emotion by student
        if len(df_filtered['Name'].unique()) > 1:
            summary.append("\nMost common emotion by student:")
            for name in df_filtered['Name'].unique():
                student_data = df_filtered[df_filtered['Name'] == name]
                most_common = student_data['Emotion'].value_counts().index[0]
                summary.append(f"  - {name}: {most_common}")
                
        plt.text(0.1, 0.5, '\n'.join(summary), fontsize=12, 
                 verticalalignment='center', horizontalalignment='left',
                 transform=plt.gca().transAxes)
                 
        # Save summary
        summary_file = os.path.join(self.analytics_dir, f"emotion_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary))
        
        # Save figure
        summary_fig_file = os.path.join(self.analytics_dir, f"emotion_summary_{timestamp}.png")
        plt.savefig(summary_fig_file)
        saved_files.append(summary_fig_file)
        
        # Show plots
        plt.show()
        
        print(f"Analytics generated and saved to {self.analytics_dir}")
        return saved_files
    
    def analyze_specific_date(self):
        """Dialog to analyze specific date"""
        # Get available dates
        if not os.path.exists(self.attendance_file):
            print("Attendance file not found")
            messagebox.showerror("Error", "Attendance file not found")
            return
            
        df = pd.read_csv(self.attendance_file)
        if df.empty:
            print("No attendance data available")
            messagebox.showerror("Error", "No attendance data available")
            return
            
        dates = sorted(df['Date'].unique())
        
        # Create date selection dialog
        root = Tk()
        root.withdraw()
        
        # Show a simple dialog with available dates
        date_str = simpledialog.askstring(
            "Select Date", 
            f"Enter date to analyze (YYYY-MM-DD):\nAvailable dates: {', '.join(dates)}",
            parent=root
        )
        
        root.destroy()
        
        if date_str:
            if date_str in dates:
                self.generate_emotion_analytics(date=date_str)
            else:
                print(f"No data available for date: {date_str}")
                messagebox.showerror("Error", f"No data available for date: {date_str}")
    
    def provide_voice_feedback(self, name, status, emotion=None):
        """
        Provide voice feedback for attendance events
        
        Parameters:
        - name: Student name
        - status: Status message (present, absent, etc.)
        - emotion: Detected emotion (optional)
        """
        if emotion:
            message = f"{name} marked {status}, current emotion: {emotion}"
        else:
            message = f"{name} marked {status}"
            
        print(f"Voice feedback: {message}")
        
        try:
            # Speak the message
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Error with voice feedback: {str(e)}")

def display_menu():
    """Display bonus features menu options."""
    print("\nBonus Features Menu")
    print("==================")
    print("1. Configure Email Notifications")
    print("2. Manage Student Email Addresses")
    print("3. Send Absence Notifications")
    print("4. Generate Emotion Analytics (All Data)")
    print("5. Analyze Specific Date")
    print("6. Test Voice Feedback")
    print("7. Back to Main Menu")
    choice = input("\nEnter your choice (1-7): ")
    return choice

def main():
    """Main function to run the bonus features module."""
    bonus_features = BonusFeatures()
    
    while True:
        choice = display_menu()
        
        if choice == '1':
            # Configure email settings
            bonus_features.configure_email()
            
        elif choice == '2':
            # Manage student emails with improved interface
            bonus_features.generate_student_emails_dialog()
            
        elif choice == '3':
            # Send absence notifications with date selection
            bonus_features.absence_notification_dialog()
            
        elif choice == '4':
            # Generate emotion analytics for all data
            bonus_features.generate_emotion_analytics()
            
        elif choice == '5':
            # Analyze specific date
            bonus_features.analyze_specific_date()
            
        elif choice == '6':
            # Test voice feedback
            name = input("Enter a name: ")
            emotion = input("Enter an emotion: ")
            bonus_features.provide_voice_feedback(name, "present", emotion)
            
        elif choice == '7':
            print("Returning to main menu.")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()