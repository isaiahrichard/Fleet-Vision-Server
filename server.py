from flask import Flask
from routes.stream_viewer import stream_viewer, start_streams
import threading
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.register_blueprint(stream_viewer)


@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    # Return a custom error page
    return "An unexpected error occurred. Please check the server logs.", 500


def start_app():
    try:
        logger.info("Starting streams...")
        start_streams()
        logger.info("Streams started successfully")
    except Exception as e:
        logger.error(f"Failed to start streams: {str(e)}")
        raise


if __name__ == "__main__":
    logger.info("Initializing server...")

    # Start the streams in a separate thread
    stream_thread = threading.Thread(target=start_app, daemon=True)
    stream_thread.start()

    # Give some time for the streams to initialize
    stream_thread.join(timeout=5)

    # Run the Flask app
    logger.info("Starting Flask app...")
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)
