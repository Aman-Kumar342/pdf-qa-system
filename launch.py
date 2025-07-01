# launch.py - Clean launch script
from main import GradioInterface

def main():
    print("🚀 Starting PDF Question-Answering System...")
    
    # Create interface
    gradio_app = GradioInterface()
    interface = gradio_app.create_interface()
    
    # Launch
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=False
    )

if __name__ == "__main__":
    main()
