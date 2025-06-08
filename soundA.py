import pandas as pd
import pyttsx3
import time
import os
import datetime

class VoiceAttendanceReader:
    """
    Reads attendance data from CSV file and provides voice feedback
    """
    
    def __init__(self, attendance_file="attendance_log.csv"):
        """
        Initialize the voice attendance reader
        
        Parameters:
        - attendance_file: CSV file containing attendance records
        """
        self.attendance_file = attendance_file
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        
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
    
    def read_attendance_data(self, specific_date=None):
        """
        Read attendance data from CSV and provide voice feedback
        
        Parameters:
        - specific_date: Optional date string (YYYY-MM-DD) to filter attendance
        """
        if not os.path.exists(self.attendance_file):
            print(f"Error: Attendance file '{self.attendance_file}' not found.")
            return
        
        try:
            # Load attendance data
            df = pd.read_csv(self.attendance_file)
            
            if df.empty:
                print("No attendance data available.")
                return
            
            # Filter by date if specified
            if specific_date:
                if specific_date in df['Date'].unique():
                    df_filtered = df[df['Date'] == specific_date]
                    print(f"Reading attendance for date: {specific_date}")
                else:
                    print(f"No attendance records found for date: {specific_date}")
                    return
            else:
                # Use most recent date
                latest_date = df['Date'].max()
                df_filtered = df[df['Date'] == latest_date]
                print(f"Reading attendance for most recent date: {latest_date}")
            
            # Speak attendance summary
            total_students = len(df_filtered)
            self.tts_engine.say(f"Reading attendance for {total_students} students.")
            self.tts_engine.runAndWait()
            time.sleep(1)  # Short pause
            
            # Read each student's attendance record
            for index, row in df_filtered.iterrows():
                name = row['Name']
                status = "present"  # Assuming records in attendance file are for present students
                emotion = row.get('Emotion', None)  # Get emotion if available
                
                # Provide voice feedback
                self.provide_voice_feedback(name, status, emotion)
                
                # Short pause between students
                time.sleep(0.5)
                
            print("Finished reading attendance data.")
            
        except Exception as e:
            print(f"Error reading attendance data: {str(e)}")
    
    def read_absent_students(self, specific_date=None):
        """
        Read a list of absent students
        
        Parameters:
        - specific_date: Optional date string (YYYY-MM-DD) to check for absences
        """
        if not os.path.exists(self.attendance_file):
            print(f"Error: Attendance file '{self.attendance_file}' not found.")
            return
        
        try:
            # Load attendance data
            df = pd.read_csv(self.attendance_file)
            
            if df.empty:
                print("No attendance data available.")
                return
            
            # Get date to check
            if specific_date:
                target_date = specific_date
            else:
                # Get most recent date in the attendance log
                target_date = df['Date'].max()
            
            # Check if we have attendance data for the target date
            if target_date not in df['Date'].unique():
                print(f"No attendance records found for {target_date}")
                return
            
            # Get list of all students
            all_students = set(df['Name'].unique())
            
            # Get students who attended on target date
            date_attendance = set(df[df['Date'] == target_date]['Name'].unique())
            
            # Identify absent students
            absent_students = all_students - date_attendance
            
            if not absent_students:
                print(f"No absent students detected for {target_date}")
                self.tts_engine.say(f"No absent students detected for {target_date}")
                self.tts_engine.runAndWait()
                return
            
            # Speak summary
            absent_count = len(absent_students)
            self.tts_engine.say(f"There are {absent_count} absent students on {target_date}")
            self.tts_engine.runAndWait()
            time.sleep(1)  # Short pause
            
            # Read each absent student
            for name in sorted(absent_students):
                self.provide_voice_feedback(name, "absent")
                time.sleep(0.5)  # Short pause between names
                
            print("Finished reading absent students.")
            
        except Exception as e:
            print(f"Error reading absent students: {str(e)}")

def display_menu():
    """Display voice reader menu options."""
    print("\nVoice Attendance Reader Menu")
    print("===========================")
    print("1. Read Today's Attendance")
    print("2. Read Attendance for Specific Date")
    print("3. List Absent Students")
    print("4. List Absent Students for Specific Date")
    print("5. Exit")
    choice = input("\nEnter your choice (1-5): ")
    return choice

def main():
    """Main function to run the voice attendance reader."""
    reader = VoiceAttendanceReader()
    
    while True:
        choice = display_menu()
        
        if choice == '1':
            # Read today's attendance
            reader.read_attendance_data()
            
        elif choice == '2':
            # Read attendance for specific date
            date = input("Enter date (YYYY-MM-DD): ")
            reader.read_attendance_data(specific_date=date)
            
        elif choice == '3':
            # List absent students
            reader.read_absent_students()
            
        elif choice == '4':
            # List absent students for specific date
            date = input("Enter date (YYYY-MM-DD): ")
            reader.read_absent_students(specific_date=date)
            
        elif choice == '5':
            print("Exiting Voice Attendance Reader.")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()