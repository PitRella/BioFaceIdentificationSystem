"""Script for user registration (console version)."""
import asyncio
import sys
from core.registration import UserRegistration
from utils.logger import setup_logger

logger = setup_logger()


def progress_callback(photo_num: int, total: int, message: str):
    """Progress callback for registration."""
    print(f"\r[{photo_num}/{total}] {message}", end="", flush=True)


async def main():
    """Main registration function."""
    print("=" * 60)
    print("User Registration")
    print("=" * 60)
    
    # Get user information
    name = input("Enter first name: ").strip()
    if not name:
        print("Error: First name is required")
        return
    
    surname = input("Enter last name: ").strip()
    if not surname:
        print("Error: Last name is required")
        return
    
    access_level_input = input("Enter access level (default: 0): ").strip()
    access_level = int(access_level_input) if access_level_input else 0
    
    print(f"\nRegistering user: {name} {surname}")
    print("Please position yourself in front of the camera.")
    print("The system will capture 5-10 photos with different angles.")
    print("Press Ctrl+C to cancel.\n")
    
    try:
        registration = UserRegistration()
        result = await registration.register_user(
            name=name,
            surname=surname,
            access_level=access_level,
            callback_progress=progress_callback
        )
        
        print("\n")  # New line after progress
        
        if result.success:
            print("=" * 60)
            print("Registration Successful!")
            print("=" * 60)
            print(f"User ID: {result.user_id}")
            print(f"Photos captured: {result.photos_captured}")
            print(f"Templates created: {result.templates_created}")
            print(f"Reference photo: {result.photo_path}")
            print("=" * 60)
        else:
            print("=" * 60)
            print("Registration Failed!")
            print("=" * 60)
            print(f"Message: {result.message}")
            if result.issues:
                print("Issues:")
                for issue in result.issues:
                    print(f"  - {issue}")
            print("=" * 60)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nRegistration cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
