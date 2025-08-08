import jwt
import datetime
import secrets

# Secret for signing (store securely, e.g., in environment variable)
from dotenv import load_dotenv
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(dir_path, ".env"))
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = os.getenv("JWT_SECRET_KEY")


# Function to generate access token
def generate_access_token(user_id: str, expires_in_minutes: int = 30):
    payload = {
        "sub": user_id,
        "exp": datetime.datetime.utcnow()
        + datetime.timedelta(minutes=expires_in_minutes),
        "iat": datetime.datetime.utcnow(),
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token


# Function to decode & verify token
def verify_access_token(token: str):
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded  # contains user_id and other claims
    except jwt.ExpiredSignatureError:
        return "Token expired"
    except jwt.InvalidTokenError:
        return "Invalid token"


def verify_access_token_fastapi(token: str = Depends(oauth2_scheme)):
    """
    Decodes and verifies a JWT access token.
    Raises an HTTPException if the token is invalid or expired.
    """
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded_token
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def generate_permanent_token_no_exp(user_id: str):
    payload = {
        "sub": user_id,
        "iat": datetime.datetime.utcnow(),
        # no 'exp' → permanent token
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token


def generate_api_token_random_length(length=40):
    return secrets.token_hex(length)


# Example usage
if __name__ == "__main__":
    token = generate_access_token("user_abc123")
    print("Access Token:", token)

    decoded = verify_access_token(token)
    print("Decoded:", decoded)

    token = generate_permanent_token_no_exp(1)
    print("Permanent Token:", token)

    # Later, validate it
    decoded = verify_access_token(token)
    print("Decoded:", decoded)
    # print("Decoded:", verify_permanent_token(token))

    api_token = generate_api_token_random_length(256)
    print("Permanent random length API Token:", api_token)
