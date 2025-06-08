import os
import re
import csv
import pandas as pd
from datetime import datetime
from tkinter import Tk, simpledialog, messagebox, StringVar, Label, Entry, Button, Frame
from tkinter import Listbox, Scrollbar, OptionMenu, filedialog
from tkinter import LEFT, RIGHT, TOP, BOTTOM, X, Y, BOTH, END, SINGLE, W, E, VERTICAL, HORIZONTAL

class EmailManager:
    """
    Enhanced email management system for student attendance system
    with improved error handling and user feedback
    """
    
    def __init__(self, attendance_file="attendance_log.csv", email_file="student_emails.csv"):
        """
        Initialize email manager with files
        
        Parameters:
        - attendance_file: Path to the attendance log CSV
        - email_file: Path to store student email addresses
        """
        self.attendance_file = attendance_file
        self.email_file = email_file
        self.students = {}
        self.load_student_emails()
    
    def is_valid_email(self, email):
        """
        Validate email format
        
        Parameters:
        - email: Email address to validate
        
        Returns:
        - Boolean: True if email is valid, False otherwise
        """
        if not email or not isinstance(email, str):
            return False
            
        email = email.strip()
        if not email:
            return False
            
        # RFC 5322 compliant email regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email))
    
    def load_student_emails(self):
        """
        Load student emails from the email file
        
        Returns:
        - Dictionary with student email information
        """
        self.students = {}
        
        # Check if email file exists and load it
        if os.path.exists(self.email_file):
            try:
                df = pd.read_csv(self.email_file)
                for _, row in df.iterrows():
                    if 'Name' in row and 'Email' in row:
                        name = row['Name']
                        email = row['Email']
                        if name and email:  # Only add if both name and email are present
                            self.students[name] = {'email': email}
                print(f"Loaded {len(self.students)} student emails from {self.email_file}")
            except Exception as e:
                print(f"Error loading student emails: {str(e)}")
        else:
            print(f"Email file {self.email_file} not found. Will create when saving.")
            
        return self.students
    
    def save_student_emails(self):
        """
        Save student emails to CSV file
        
        Returns:
        - Boolean: True if saved successfully, False otherwise
        """
        try:
            # Create data structure for saving
            email_data = []
            for name, info in self.students.items():
                email = info.get('email', '').strip()
                if email:  # Only save if email is provided
                    email_data.append({'Name': name, 'Email': email})
            
            # Save to CSV
            df = pd.DataFrame(email_data)
            df.to_csv(self.email_file, index=False)
            print(f"Saved {len(email_data)} email addresses to {self.email_file}")
            return True
        except Exception as e:
            print(f"Error saving student emails: {str(e)}")
            return False
    
    def get_unique_students(self):
        """
        Get list of unique student names from attendance file
        
        Returns:
        - List of student names
        """
        if not os.path.exists(self.attendance_file):
            print(f"Attendance file {self.attendance_file} not found")
            return []
            
        try:
            df = pd.read_csv(self.attendance_file)
            return sorted(df['Name'].unique())
        except Exception as e:
            print(f"Error reading attendance file: {str(e)}")
            return []
    
    def update_student_email(self, name, email):
        """
        Update or add a student email
        
        Parameters:
        - name: Student name
        - email: Email address
        
        Returns:
        - Boolean: True if updated, False if invalid
        """
        if not name:
            return False
            
        if email and not self.is_valid_email(email):
            return False
            
        if name not in self.students:
            self.students[name] = {}
            
        self.students[name]['email'] = email.strip()
        return True
    
    def remove_student_email(self, name):
        """
        Remove a student's email
        
        Parameters:
        - name: Student name
        
        Returns:
        - Boolean: True if removed, False otherwise
        """
        if name in self.students:
            if 'email' in self.students[name]:
                del self.students[name]['email']
            if not self.students[name]:  # If no other data, remove entirely
                del self.students[name]
            return True
        return False
    
    def bulk_import_emails(self, file_path):
        """
        Import emails from CSV file
        
        Parameters:
        - file_path: Path to CSV file with Name and Email columns
        
        Returns:
        - count: Number of emails imported
        """
        if not os.path.exists(file_path):
            return 0
            
        try:
            df = pd.read_csv(file_path)
            
            if 'Name' not in df.columns or 'Email' not in df.columns:
                return 0
                
            count = 0
            for _, row in df.iterrows():
                name = row['Name']
                email = row['Email']
                
                if name and email and self.is_valid_email(email):
                    if name not in self.students:
                        self.students[name] = {}
                    self.students[name]['email'] = email.strip()
                    count += 1
                    
            return count
        except Exception as e:
            print(f"Error importing emails: {str(e)}")
            return 0
    
    def manage_student_emails_dialog(self):
        """
        Display a dialog to manage student email addresses
        """
        # Load unique students from attendance file
        unique_names = self.get_unique_students()
        
        if not unique_names:
            messagebox.showerror("Error", "No students found in attendance records")
            return
        
        # Create UI
        root = Tk()
        root.title("Student Email Manager")
        root.geometry("700x550")
        
        # Main frame
        main_frame = Frame(root, padx=20, pady=20)
        main_frame.pack(fill=BOTH, expand=True)
        
        # Header
        header_frame = Frame(main_frame)
        header_frame.pack(fill=X, pady=(0, 15))
        
        Label(header_frame, text="Student Email Manager", font=("Arial", 16, "bold")).pack(anchor=W)
        Label(header_frame, text="Manage email addresses for attendance notifications", font=("Arial", 10)).pack(anchor=W)
        
        # Student count information
        info_frame = Frame(main_frame)
        info_frame.pack(fill=X, pady=(0, 10))
        
        # Status counts
        status_info = StringVar()
        status_info.set(f"Total students: {len(unique_names)} | Emails configured: {len(self.students)}")
        Label(info_frame, textvariable=status_info, fg="blue").pack(anchor=W)
        
        # Content frame (left: list, right: details)
        content_frame = Frame(main_frame)
        content_frame.pack(fill=BOTH, expand=True, pady=10)
        
        # Left side: List with search
        list_frame = Frame(content_frame, width=300)
        list_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))
        
        # Search frame
        search_frame = Frame(list_frame)
        search_frame.pack(fill=X, pady=(0, 5))
        
        Label(search_frame, text="Search:").pack(side=LEFT)
        search_var = StringVar()
        search_entry = Entry(search_frame, textvariable=search_var, width=25)
        search_entry.pack(side=LEFT, padx=5, fill=X, expand=True)
        
        # Filter dropdown
        filter_frame = Frame(list_frame)
        filter_frame.pack(fill=X, pady=(0, 5))
        
        Label(filter_frame, text="Show:").pack(side=LEFT)
        filter_var = StringVar(value="All Students")
        filter_options = ["All Students", "With Email", "Without Email"]
        filter_menu = OptionMenu(filter_frame, filter_var, *filter_options)
        filter_menu.pack(side=LEFT, padx=5)
        
        # Student list
        list_container = Frame(list_frame)
        list_container.pack(fill=BOTH, expand=True)
        
        y_scrollbar = Scrollbar(list_container, orient=VERTICAL)
        y_scrollbar.pack(side=RIGHT, fill=Y)
        
        student_listbox = Listbox(list_container, selectmode=SINGLE, yscrollcommand=y_scrollbar.set)
        student_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        y_scrollbar.config(command=student_listbox.yview)
        
        # Right side: Details
        details_frame = Frame(content_frame, width=350)
        details_frame.pack(side=RIGHT, fill=BOTH, expand=True)
        
        # Student details
        student_frame = Frame(details_frame, pady=10)
        student_frame.pack(fill=X)
        
        # Student name
        name_frame = Frame(student_frame)
        name_frame.pack(fill=X, pady=5)
        
        Label(name_frame, text="Student:").grid(row=0, column=0, sticky=W)
        name_var = StringVar()
        name_label = Label(name_frame, textvariable=name_var, font=("Arial", 12, "bold"))
        name_label.grid(row=0, column=1, sticky=W, padx=5)
        
        # Email field
        email_frame = Frame(student_frame)
        email_frame.pack(fill=X, pady=5)
        
        Label(email_frame, text="Email:").grid(row=0, column=0, sticky=W)
        email_var = StringVar()
        email_entry = Entry(email_frame, textvariable=email_var, width=40)
        email_entry.grid(row=0, column=1, sticky=W+E, padx=5)
        
        # Email status
        email_status_var = StringVar()
        email_status_label = Label(student_frame, textvariable=email_status_var, fg="red")
        email_status_label.pack(anchor=W, pady=(0, 5))
        
        # Buttons for individual student
        button_frame = Frame(student_frame)
        button_frame.pack(fill=X, pady=5)
        
        # Function to update email status
        def update_email_validity(*args):
            email = email_var.get().strip()
            if not email:
                email_status_var.set("")
                email_entry.config(bg="white")
            elif self.is_valid_email(email):
                email_status_var.set("Valid email format")
                email_entry.config(bg="#e0f7e0")  # Light green
            else:
                email_status_var.set("Invalid email format")
                email_entry.config(bg="#f7e0e0")  # Light red
                
        email_var.trace("w", update_email_validity)
        
        # Function to update student email
        def update_student():
            selected_indices = student_listbox.curselection()
            if not selected_indices:
                messagebox.showerror("Error", "No student selected")
                return
                
            name = student_listbox.get(selected_indices[0])
            email = email_var.get().strip()
            
            if email and not self.is_valid_email(email):
                messagebox.showerror("Error", "Invalid email format")
                return
                
            if self.update_student_email(name, email):
                messagebox.showinfo("Success", f"Updated email for {name}")
                refresh_list()
                
        # Function to clear email
        def clear_email():
            selected_indices = student_listbox.curselection()
            if not selected_indices:
                return
                
            name = student_listbox.get(selected_indices[0])
            email_var.set("")
            
            if self.remove_student_email(name):
                messagebox.showinfo("Success", f"Cleared email for {name}")
                refresh_list()
        
        # Add update and clear buttons
        Button(button_frame, text="Update", command=update_student, width=15).pack(side=LEFT, padx=5)
        Button(button_frame, text="Clear", command=clear_email, width=15).pack(side=LEFT, padx=5)
        
        # Bulk operations frame
        bulk_frame = Frame(details_frame, pady=10, relief="groove", bd=1)
        bulk_frame.pack(fill=X, pady=10)
        
        Label(bulk_frame, text="Bulk Operations", font=("Arial", 10, "bold")).pack(anchor=W, padx=10, pady=5)
        
        bulk_buttons = Frame(bulk_frame, padx=10, pady=5)
        bulk_buttons.pack(fill=X)
        
        # Import function
        def import_csv():
            file_path = filedialog.askopenfilename(
                title="Select CSV File",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                parent=root
            )
            
            if not file_path:
                return
                
            count = self.bulk_import_emails(file_path)
            
            if count > 0:
                messagebox.showinfo("Import Complete", f"Successfully imported {count} email addresses")
                refresh_list()
            else:
                messagebox.showerror("Import Failed", "No valid email addresses found or file format incorrect")
        
        # Export function
        def export_csv():
            if not self.students:
                messagebox.showinfo("Export", "No email addresses to export")
                return
                
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"student_emails_export_{timestamp}.csv"
            
            file_path = filedialog.asksaveasfilename(
                title="Save Email Export",
                filetypes=[("CSV files", "*.csv")],
                defaultextension=".csv",
                initialfile=default_filename,
                parent=root
            )
            
            if not file_path:
                return
                
            try:
                # Create data for export
                email_data = []
                for name, info in self.students.items():
                    email = info.get('email', '').strip()
                    if email:  # Only export if email is provided
                        email_data.append({'Name': name, 'Email': email})
                
                # Save to CSV
                pd.DataFrame(email_data).to_csv(file_path, index=False)
                messagebox.showinfo("Export Complete", f"Exported {len(email_data)} email addresses")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
        
        # Add bulk operations buttons
        Button(bulk_buttons, text="Import CSV", command=import_csv, width=15).pack(side=LEFT, padx=5)
        Button(bulk_buttons, text="Export CSV", command=export_csv, width=15).pack(side=LEFT, padx=5)
        
        # Email validation status
        validation_frame = Frame(details_frame, pady=10)
        validation_frame.pack(fill=X, pady=10)
        
        # Add validation summary
        validation_var = StringVar()
        validation_label = Label(validation_frame, textvariable=validation_var, fg="blue", wraplength=300, justify=LEFT)
        validation_label.pack(anchor=W, fill=X)
        
        # Bottom buttons
        bottom_frame = Frame(main_frame)
        bottom_frame.pack(fill=X, pady=10)
        
        # Function to save all and exit
        def save_and_exit():
            if self.save_student_emails():
                messagebox.showinfo("Success", f"Saved {len(self.students)} student email records")
                root.destroy()
            else:
                messagebox.showerror("Error", "Failed to save student emails")
        
        # Add save/cancel buttons
        Button(bottom_frame, text="Save & Exit", command=save_and_exit, width=15).pack(side=RIGHT, padx=5)
        Button(bottom_frame, text="Cancel", command=root.destroy, width=15).pack(side=RIGHT, padx=5)
        
        # Functions to populate and filter the list
        def refresh_list():
            # Update status info first
            valid_emails = sum(1 for info in self.students.values() if info.get('email', ''))
            status_info.set(f"Total students: {len(unique_names)} | Emails configured: {valid_emails}")
            
            # Update validation summary
            invalid_emails = []
            for name, info in self.students.items():
                email = info.get('email', '')
                if email and not self.is_valid_email(email):
                    invalid_emails.append(name)
            
            if invalid_emails:
                validation_var.set(f"Warning: {len(invalid_emails)} student(s) have invalid email formats")
            else:
                validation_var.set("")
            
            # Now update the listbox
            student_listbox.delete(0, END)
            
            filter_type = filter_var.get()
            search_text = search_var.get().lower()
            
            for name in unique_names:
                # Apply filter
                if filter_type == "With Email" and name not in self.students:
                    continue
                if filter_type == "Without Email" and name in self.students and self.students[name].get('email'):
                    continue
                
                # Apply search
                if search_text and search_text not in name.lower():
                    continue
                
                # Add to listbox
                if name in self.students and self.students[name].get('email'):
                    # Show with email icon
                    student_listbox.insert(END, f"{name} ✉")
                else:
                    student_listbox.insert(END, name)
        
        # Function to handle selection change
        def on_select(event):
            try:
                index = student_listbox.curselection()[0]
                selected_name = student_listbox.get(index)
                
                # Remove email icon if present
                if " ✉" in selected_name:
                    selected_name = selected_name.replace(" ✉", "")
                
                name_var.set(selected_name)
                
                # Set email if exists
                if selected_name in self.students:
                    email_var.set(self.students[selected_name].get('email', ''))
                else:
                    email_var.set('')
                    
                # Reset status
                email_status_var.set("")
                update_email_validity()
                
            except IndexError:
                pass
        
        # Bind events
        student_listbox.bind('<<ListboxSelect>>', on_select)
        search_var.trace("w", lambda *args: refresh_list())
        filter_var.trace("w", lambda *args: refresh_list())
        
        # Initial list population
        refresh_list()
        
        # Make dialog modal
        root.transient()
        root.grab_set()
        root.focus_set()
        
        # Center on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Search field gets focus
        search_entry.focus_set()
        
        root.mainloop()

# Usage example
if __name__ == "__main__":
    manager = EmailManager()
    manager.manage_student_emails_dialog()