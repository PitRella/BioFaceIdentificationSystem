"""Script for face identification (console version)."""
import asyncio
import sys
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.identification import FaceIdentification
from core.video_capture import VideoCapture
from config import FRAME_SKIP
from utils.logger import setup_logger

logger = setup_logger()


async def main():
    """Main identification function."""
    print("=" * 60)
    print("Face Identification")
    print("=" * 60)
    print("Press 'q' to quit")
    print("=" * 60)
    
    try:
        # Initialize identification
        identification = FaceIdentification()
        
        # Load descriptors cache
        print("Loading descriptors from database...")
        count = await identification.load_descriptors_cache()
        print(f"Loaded {count} descriptors")
        
        if count == 0:
            print("Error: No users registered in database")
            print("Please register users first using: python scripts/register_user.py")
            sys.exit(1)
        
        print("\nStarting identification...")
        print("Position yourself in front of the camera.\n")
        
        frame_count = 0
        
        with VideoCapture() as cap:
            if not cap.is_opened():
                print("Error: Failed to open camera")
                sys.exit(1)
            
            while True:
                frame = cap.read()
                if frame is None:
                    continue
                
                frame_count += 1
                
                # Process every Nth frame
                if frame_count % FRAME_SKIP != 0:
                    continue
                
                # Identify faces
                results = await identification.identify_frame(frame)
                
                # Draw results on frame
                display_frame = frame.copy()
                
                for result in results:
                    if result.success:
                        # Draw bounding box (we need face location, but for now just show text)
                        text = f"{result.user_name} {result.user_surname} ({result.confidence:.0%})"
                        color = (0, 255, 0)  # Green
                    else:
                        text = "Unknown"
                        color = (0, 0, 255)  # Red
                    
                    # Put text on frame
                    cv2.putText(
                        display_frame,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2
                    )
                
                # Show frame
                cv2.imshow("Face Identification", display_frame)
                
                # Print results to console
                if results:
                    for result in results:
                        if result.success:
                            print(f"✓ Identified: {result.user_name} {result.user_surname} "
                                  f"(ID: {result.user_id}, Confidence: {result.confidence:.2%})")
                        else:
                            print("✗ Unknown person")
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        print("\nIdentification stopped")
        
    except KeyboardInterrupt:
        print("\n\nIdentification interrupted by user")
    except Exception as e:
        logger.error(f"Identification error: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
