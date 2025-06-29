#!/usr/bin/env python3
"""
MeshBuilder - Photorealistic 3D Reconstruction Application
Main entry point for the application
"""
import os
import sys
import logging
import traceback
from pathlib import Path

# Add project directories to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))
sys.path.insert(0, str(project_root / "frontend"))
sys.path.insert(0, str(project_root / "interface"))
sys.path.insert(0, str(project_root / "licensing"))

# Import PyQt6 components
from PyQt6.QtWidgets import QApplication, QMessageBox, QSplashScreen
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QIcon

# Import application components
from frontend.Frontend_window import MeshbuilderMainWindow
from licensing.licence_system import LicenceSystem, check_licence_and_activate


# Configure logging
def setup_logging():
    """Set up application logging"""
    log_dir = Path.home() / ".meshbuilder" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "meshbuilder.log"
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log startup
    logging.info("="*60)
    logging.info("MeshBuilder Application Starting")
    logging.info("="*60)


def show_splash_screen(app):
    """Show splash screen during startup"""
    splash_pixmap = QPixmap(600, 400)
    splash_pixmap.fill(Qt.GlobalColor.black)
    
    # Try to load logo
    logo_path = project_root / "assets" / "icon 1c_1.png"
    if logo_path.exists():
        logo = QPixmap(str(logo_path))
        splash_pixmap = logo.scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio, 
                                   Qt.TransformationMode.SmoothTransformation)
    
    splash = QSplashScreen(splash_pixmap)
    splash.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
    splash.show()
    
    # Show messages
    splash.showMessage("Loading MeshBuilder...", 
                      Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, 
                      Qt.GlobalColor.white)
    
    app.processEvents()
    
    return splash


def check_dependencies():
    """Check if required dependencies are installed"""
    missing_deps = []
    
    # Check critical dependencies
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import trimesh
    except ImportError:
        logging.warning("trimesh not available - mesh operations will be limited")
    
    try:
        import open3d
    except ImportError:
        logging.warning("open3d not available - point cloud operations will be limited")
    
    if missing_deps:
        error_msg = (
            f"Critical dependencies missing: {', '.join(missing_deps)}\n\n"
            "Please install them using:\n"
            f"pip install {' '.join(missing_deps)}"
        )
        QMessageBox.critical(None, "Missing Dependencies", error_msg)
        return False
    
    return True


def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    # Log the exception
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    # Show error dialog
    error_msg = f"{exc_type.__name__}: {exc_value}"
    detailed_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Icon.Critical)
    msg_box.setWindowTitle("Application Error")
    msg_box.setText("An unexpected error occurred:")
    msg_box.setInformativeText(error_msg)
    msg_box.setDetailedText(detailed_msg)
    msg_box.exec()


def main():
    """Main application entry point"""
    # Set up logging first
    setup_logging()
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("MeshBuilder")
    app.setApplicationDisplayName("MeshBuilder - 3D Reconstruction")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("Immersive Engineering")
    app.setOrganizationDomain("immersive-engineering.com")
    
    # Set application icon
    icon_path = project_root / "assets" / "icon 1c_1.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    # Install exception handler
    sys.excepthook = handle_exception
    
    # Show splash screen
    splash = show_splash_screen(app)
    
    try:
        # Update splash
        splash.showMessage("Checking dependencies...", 
                          Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, 
                          Qt.GlobalColor.white)
        app.processEvents()
        
        # Check dependencies
        if not check_dependencies():
            splash.close()
            return 1
        
        # Update splash
        splash.showMessage("Validating license...", 
                          Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, 
                          Qt.GlobalColor.white)
        app.processEvents()
        
        # Check license
        app_id = "MTc1NzQxNDU4MC1iOGExOWVjMDE2YjAxZWUw"  # Your app ID
        
        # Initialize license system
        licence_system = LicenceSystem(app_id=app_id)
        
        # Check if already licensed
        if not licence_system.is_licenced():
            splash.close()
            
            # Show license activation
            logging.info("No valid license found, showing activation UI")
            if not check_licence_and_activate(app_id=app_id):
                logging.info("License activation cancelled or failed")
                QMessageBox.warning(
                    None, 
                    "License Required", 
                    "A valid license is required to use MeshBuilder.\n"
                    "Please contact support for a license key."
                )
                return 1
        else:
            # Log license info
            license_info = licence_system.get_licence_info()
            if license_info:
                days_remaining = licence_system.get_days_remaining()
                logging.info(f"License valid, {days_remaining} days remaining")
                
                # Show warning if expiring soon
                if days_remaining and days_remaining < 30:
                    splash.close()
                    QMessageBox.warning(
                        None,
                        "License Expiring Soon",
                        f"Your license will expire in {days_remaining} days.\n"
                        "Please contact support to renew."
                    )
        
        # Update splash
        splash.showMessage("Loading application...", 
                          Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, 
                          Qt.GlobalColor.white)
        app.processEvents()
        
        # Create main window
        window = MeshbuilderMainWindow()
        
        # Configure window
        window.setWindowTitle("MeshBuilder - Photorealistic 3D Reconstruction")
        window.setMinimumSize(1024, 768)
        
        # Center window on screen
        screen = app.primaryScreen()
        screen_rect = screen.availableGeometry()
        window_rect = window.frameGeometry()
        window_rect.moveCenter(screen_rect.center())
        window.move(window_rect.topLeft())
        
        # Close splash and show window
        splash.finish(window)
        window.show()
        
        # Log successful startup
        logging.info("MeshBuilder application started successfully")
        
        # Add startup message
        if hasattr(window, 'add_log_message'):
            window.add_log_message("MeshBuilder initialized successfully")
            window.add_log_message(f"License valid for {licence_system.get_days_remaining()} days")
            
            # Show system info
            try:
                import psutil
                memory_gb = psutil.virtual_memory().total / (1024**3)
                window.add_log_message(f"System: {memory_gb:.1f} GB RAM, {psutil.cpu_count()} CPU cores")
            except:
                pass
        
        # Run application
        exit_code = app.exec()
        
        # Cleanup
        logging.info("MeshBuilder application shutting down")
        
        return exit_code
        
    except Exception as e:
        logging.error(f"Fatal error during startup: {e}", exc_info=True)
        splash.close()
        
        QMessageBox.critical(
            None,
            "Startup Error",
            f"Failed to start MeshBuilder:\n{str(e)}"
        )
        
        return 1


if __name__ == "__main__":
    sys.exit(main())