# deploy.py - Production deployment script
import sys
import os
from config import Config
from main import GradioInterface

def check_system_requirements():
    """Check if system meets requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
        
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 4:
            print(f"⚠️  Low memory: {memory_gb:.1f}GB (4GB+ recommended)")
        else:
            print(f"✅ Memory: {memory_gb:.1f}GB")
    except:
        print("⚠️  Could not check memory")
        
    # Check disk space
    try:
        disk_space = psutil.disk_usage('.').free / (1024**3)
        if disk_space < 2:
            print(f"⚠️  Low disk space: {disk_space:.1f}GB")
        else:
            print(f"✅ Disk space: {disk_space:.1f}GB available")
    except:
        print("⚠️  Could not check disk space")
        
    return True

def setup_environment():
    """Setup deployment environment"""
    print("🛠️  Setting up environment...")
    
    # Create directories
    Config.create_directories()
    print("✅ Directories created")
    
    # Set environment variables
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid warnings
    print("✅ Environment configured")

def deploy_system():
    """Deploy the PDF QA system"""
    print("🚀 Deploying PDF Question-Answering System...")
    print("="*60)
    
    if not check_system_requirements():
        print("❌ System requirements not met")
        return False
        
    setup_environment()
    
    try:
        # Initialize system
        print("📚 Initializing PDF QA System...")
        gradio_app = GradioInterface()
        
        # Choose interface (enhanced or basic)
        use_enhanced = input("Use enhanced interface? (y/n): ").lower().startswith('y')
        
        if use_enhanced:
            interface = gradio_app.create_enhanced_interface()
            print("✅ Enhanced interface created")
        else:
            interface = gradio_app.create_interface()
            print("✅ Basic interface created")
        
        # Launch configuration
        share_public = input("Create public link? (y/n): ").lower().startswith('y')
        
        print(f"\n🌐 Launching on http://{Config.HOST}:{Config.PORT}")
        if share_public:
            print("🔗 Public link will be generated")
            
        # Launch system
        interface.launch(
            server_name=Config.HOST,
            server_port=Config.PORT,
            share=share_public,
            debug=Config.DEBUG,
            show_error=True,
            favicon_path=None,
            ssl_verify=False
        )
        
    except KeyboardInterrupt:
        print("\n👋 System shutdown requested")
        return True
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        return False

if __name__ == "__main__":
    deploy_system()
