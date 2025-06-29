"""
License system for MeshBuilder
This code is based on the original license_system.py with modifications to support
the MeshBuilder application
"""
import os
import platform
import hashlib
import base64
import json
import datetime
import hmac
import time
import logging
import tkinter as tk
from tkinter import messagebox, simpledialog

# Set up logger
logger = logging.getLogger(__name__)

class LicenceSystem:
    def __init__(self, app_id="MTc1NzQxNDU4MC1iOGExOWVjMDE2YjAxZWUw", licence_path=None):
        """
        Initialize the licence system
        
        Parameters:
        - app_id: Your application ID (must match in generator)
        - licence_path: Optional custom path for licence storage
        """
        # Application ID
        self.app_id = app_id
        
        # Path for licence storage
        self.licence_path = licence_path or os.path.join(
            os.path.expanduser("~"),
            ".meshbuilder",
            "licence_data.json"
        )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.licence_path), exist_ok=True)
        
        # Secret key for HMAC verification
        self.verification_key = b"PUBLIC_VERIFICATION_KEY_2025"
        
        # Current licence info
        self.current_licence = None
        
        # Load existing licence if available
        self._load_licence()
        
        logger.info(f"License system initialized with app_id: {app_id}")
    
    def generate_device_id(self):
        """Generate a unique but consistent device ID using minimal system info"""
        system_info = platform.system() + platform.node() + platform.processor()
        
        # Get disk info - OS specific
        if platform.system() == 'Windows':
            try:
                import ctypes
                volume_info = ctypes.create_string_buffer(64)
                ctypes.windll.kernel32.GetVolumeInformationA(
                    b"C:\\", None, 0, volume_info, None, None, None, 0
                )
                vol_serial = str(int.from_bytes(volume_info.raw[:4], byteorder='little'))
                system_info += vol_serial
            except:
                logger.warning("Failed to get Windows volume information")
                pass
        else:
            try:
                if os.path.exists('/etc/machine-id'):
                    with open('/etc/machine-id', 'r') as f:
                        system_info += f.read().strip()
            except:
                logger.warning("Failed to get Linux machine ID")
                pass
        
        # Create a hash of the system info
        device_id = hashlib.sha256(system_info.encode()).hexdigest()
        logger.debug(f"Generated device ID: {device_id[:8]}...")
        return device_id
    
    def generate_challenge_code(self):
        """Generate a challenge code based on device ID"""
        device_id = self.generate_device_id()
        timestamp = str(int(time.time()))  # Unix timestamp
        
        # Create challenge data
        challenge_data = f"{device_id}|{timestamp}|{self.app_id}"
        
        # Base64 encode for readability
        challenge_code = base64.b64encode(challenge_data.encode()).decode()
        logger.info("Challenge code generated")
        return challenge_code
    
    def validate_licence(self, licence_key):
        """
        Validate the provided licence key
        
        Returns:
        - (is_valid, message): Tuple with validation result and message
        """
        try:
            # Decode the base64 licence key
            decoded_data = base64.b64decode(licence_key).decode()
            
            # Split the licence data and signature
            parts = decoded_data.split('.')
            if len(parts) != 2:
                logger.warning("Invalid license format (missing signature delimiter)")
                return False, "Invalid licence format"
                
            licence_data = parts[0]
            signature = parts[1]
            
            # Verify signature with public verification key
            computed_sig = hmac.new(
                self.verification_key, 
                licence_data.encode(), 
                hashlib.sha256
            ).hexdigest()
            
            if computed_sig != signature:
                logger.warning("Invalid license signature")
                return False, "Invalid licence signature"
            
            # Parse licence components
            licence_parts = licence_data.split('|')
            if len(licence_parts) != 4:
                logger.warning("Invalid license data format (incorrect parts)")
                return False, "Invalid licence data format"
                
            licence_device_id = licence_parts[0]
            expiration_timestamp = licence_parts[1]
            app_id = licence_parts[2]
            features = licence_parts[3]
            
            # Check device ID
            current_device_id = self.generate_device_id()
            if licence_device_id != current_device_id:
                logger.warning(f"License device ID mismatch - expected: {licence_device_id[:8]}..., got: {current_device_id[:8]}...")
                return False, "This licence is for a different device"
            
            # Check app ID
            if app_id != self.app_id:
                logger.warning(f"License app ID mismatch - expected: {self.app_id}, got: {app_id}")
                return False, "This licence is for a different application"
            
            # Check expiration
            expiration_date = datetime.datetime.fromtimestamp(int(expiration_timestamp))
            if datetime.datetime.now() > expiration_date:
                logger.warning(f"License expired on {expiration_date.strftime('%Y-%m-%d')}")
                return False, f"Licence expired on {expiration_date.strftime('%Y-%m-%d')}"
            
            # licence is valid, save it
            self._save_licence(licence_key, expiration_date.isoformat(), features)
            
            logger.info(f"License validated successfully, valid until {expiration_date.strftime('%Y-%m-%d')}")
            return True, f"Licence valid until {expiration_date.strftime('%Y-%m-%d')}"
            
        except Exception as e:
            logger.error(f"Error validating license: {str(e)}")
            return False, f"Invalid licence key: {str(e)}"
    
    def _save_licence(self, licence_key, expiration_date, features):
        """Save the licence data to a file (internal use)"""
        data = {
            "licence_key": licence_key,
            "expiration_date": expiration_date,
            "features": features,
            "activation_date": datetime.datetime.now().isoformat()
        }
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.licence_path)), exist_ok=True)
            
            with open(self.licence_path, "w") as f:
                json.dump(data, f)
                
            # Update current licence
            self.current_licence = data
            logger.info(f"License saved to {self.licence_path}")
        except Exception as e:
            logger.error(f"Error saving licence: {e}")
    
    def _load_licence(self):
        """Load saved licence if available (internal use)"""
        try:
            if os.path.exists(self.licence_path):
                with open(self.licence_path, "r") as f:
                    data = json.load(f)
                    self.current_licence = data
                logger.info("License loaded from file")
            else:
                logger.info("No license file found")
        except Exception as e:
            logger.error(f"Error loading licence: {e}")
            self.current_licence = None
    
    def is_licenced(self):
        """Check if application is licenced"""
        if not self.current_licence:
            logger.info("No current license found")
            return False
            
        # Validate the saved licence key
        is_valid, _ = self.validate_licence(self.current_licence["licence_key"])
        return is_valid
    
    def get_licence_info(self):
        """
        Get licence information
        
        Returns:
        - Dictionary with licence details or None if no valid licence
        """
        if not self.is_licenced():
            return None
            
        # Return a copy to prevent modification
        return dict(self.current_licence)
    
    def get_expiration_date(self, format="%Y-%m-%d"):
        """Get formatted expiration date string or None if no licence"""
        if not self.current_licence:
            return None
            
        try:
            exp_date = datetime.datetime.fromisoformat(self.current_licence["expiration_date"])
            return exp_date.strftime(format)
        except:
            logger.error("Error formatting expiration date")
            return None
    
    def get_days_remaining(self):
        """Get days remaining until licence expires, or None if no licence"""
        if not self.current_licence:
            return None
            
        try:
            exp_date = datetime.datetime.fromisoformat(self.current_licence["expiration_date"])
            today = datetime.datetime.now()
            return (exp_date - today).days
        except:
            logger.error("Error calculating days remaining")
            return None
    
    def get_licence_type(self):
        """Get licence type/features string or None if no licence"""
        if not self.current_licence:
            return None
            
        return self.current_licence.get("features", "STANDARD")


# UI Components for licence activation
class LicenceActivationUI:
    def __init__(self, licence_system, parent=None, callback=None):
        """
        Initialize licence activation UI
        
        Parameters:
        - licence_system: Instance of LicenceSystem
        - parent: Parent tkinter window (optional)
        - callback: Function to call after successful activation
        """
        self.licence_system = licence_system
        self.parent = parent
        self.callback = callback
        
    def show_activation_screen(self, message=""):
        """Show licence activation dialog"""
        # Create a new window if no parent provided
        if not self.parent:
            root = tk.Tk()
            root.title("Licence Activation")
            root.geometry("500x300")
            
            # Don't destroy when closed
            def on_close():
                root.withdraw()
            root.protocol("WM_DELETE_WINDOW", on_close)
            
            self.dialog = root
        else:
            # Create as dialog
            self.dialog = tk.Toplevel(self.parent)
            self.dialog.title("Licence Activation")
            self.dialog.geometry("500x300")
            self.dialog.transient(self.parent)
            self.dialog.grab_set()
        
        # Configure style
        self.dialog.configure(bg="#333333")
        
        # Title
        title_label = tk.Label(
            self.dialog, 
            text="Software Activation Required", 
            font=("Arial", 16),
            fg="white",
            bg="#333333"
        )
        title_label.pack(pady=20)
        
        # Message
        message_label = tk.Label(
            self.dialog, 
            text=message, 
            font=("Arial", 10),
            fg="#FF9900",
            bg="#333333",
            wraplength=400
        )
        message_label.pack(pady=10)
        
        # Challenge code button
        challenge_button = tk.Button(
            self.dialog,
            text="Generate Challenge Code",
            command=self.show_challenge_code,
            bg="#555555",
            fg="white",
            padx=10,
            pady=5
        )
        challenge_button.pack(pady=10)
        
        # licence entry
        tk.Label(
            self.dialog, 
            text="Enter Licence Key:", 
            fg="white",
            bg="#333333"
        ).pack(pady=(10, 0))
        
        licence_frame = tk.Frame(self.dialog, bg="#333333")
        licence_frame.pack(pady=5)
        
        self.licence_entry = tk.Entry(licence_frame, width=40)
        self.licence_entry.pack(side=tk.LEFT, padx=5)
        
        activate_button = tk.Button(
            licence_frame,
            text="Activate",
            command=self.activate_licence,
            bg="#4CAF50",
            fg="white"
        )
        activate_button.pack(side=tk.LEFT)
        
        logger.info("License activation UI initialized")
        self.dialog.mainloop()
    
    def show_challenge_code(self):
        """Show dialog with challenge code"""
        # Generate and display the challenge code
        challenge_code = self.licence_system.generate_challenge_code()
        
        # Create dialog to display and copy the code
        code_dialog = tk.Toplevel(self.dialog)
        code_dialog.title("Challenge Code")
        code_dialog.geometry("500x200")
        code_dialog.configure(bg="#333333")
        code_dialog.transient(self.dialog)
        code_dialog.grab_set()
        
        tk.Label(
            code_dialog, 
            text="Your Challenge Code:", 
            font=("Arial", 12),
            fg="white",
            bg="#333333"
        ).pack(pady=(20, 10))
        
        code_entry = tk.Entry(code_dialog, width=50)
        code_entry.insert(0, challenge_code)
        code_entry.pack(pady=10)
        
        # Copy button
        def copy_to_clipboard():
            code_dialog.clipboard_clear()
            code_dialog.clipboard_append(challenge_code)
            tk.Label(
                code_dialog, 
                text="Copied to clipboard!", 
                fg="green",
                bg="#333333"
            ).pack(pady=5)
        
        copy_button = tk.Button(
            code_dialog,
            text="Copy to Clipboard",
            command=copy_to_clipboard,
            bg="#555555",
            fg="white"
        )
        copy_button.pack(pady=10)
        
        # Instructions
        tk.Label(
            code_dialog, 
            text="Send this code to the software provider to get your licence key.", 
            fg="white",
            bg="#333333",
            wraplength=400
        ).pack(pady=10)
        
        logger.info(f"Challenge code displayed to user: {challenge_code[:10]}...")
    
    def activate_licence(self):
        """Validate and activate licence"""
        # Get the licence key from entry
        licence_key = self.licence_entry.get().strip()
        
        if not licence_key:
            messagebox.showerror("Error", "Please enter a licence key.")
            return
        
        # Validate the licence
        is_valid, message = self.licence_system.validate_licence(licence_key)
        
        if is_valid:
            # Show success
            messagebox.showinfo("Success", message)
            
            # Call callback if provided
            if self.callback:
                self.callback()
                
            # Close the dialog
            self.dialog.destroy()
            logger.info("License activation successful")
        else:
            # Show error
            messagebox.showerror("Invalid Licence", message)
            logger.warning(f"License activation failed: {message}")


# Utility functions for easy integration
def check_licence_and_activate(app_id="MTc1NzQxNDU4MC1iOGExOWVjMDE2YjAxZWUw", parent=None, success_callback=None):
    """
    Check licence and show activation UI if needed
    
    Parameters:
    - app_id: Your application ID
    - parent: Parent tkinter window (optional)
    - success_callback: Function to call after successful activation
    
    Returns:
    - True if licenced, False if not licenced and user canceled activation
    """
    logger.info("Checking license...")
    
    # Initialize licence system
    licence_system = LicenceSystem(app_id=app_id)
    
    # If already licenced, return True
    if licence_system.is_licenced():
        logger.info("Valid license found")
        return True
    
    logger.info("No valid license found, showing activation UI")
    
    # Show activation UI
    activator = LicenceActivationUI(
        licence_system=licence_system,
        parent=parent,
        callback=success_callback
    )
    
    activator.show_activation_screen("Please activate the software to continue.")
    
    # Check again after activation UI closes
    result = licence_system.is_licenced()
    logger.info(f"License check result after activation attempt: {result}")
    return result


# Example of how to use in a standalone script
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the module
    licence_system = LicenceSystem()
    
    if licence_system.is_licenced():
        print(f"Licenced until: {licence_system.get_expiration_date()}")
        print(f"Days remaining: {licence_system.get_days_remaining()}")
        print(f"Licence type: {licence_system.get_licence_type()}")
    else:
        print("Not licenced. Starting activation...")
        LicenceActivationUI(licence_system).show_activation_screen()