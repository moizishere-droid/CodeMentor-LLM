"""Password hashing and revocable, expiring account sessions."""
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.database import AuthSession, User, get_db

router = APIRouter(prefix="/auth", tags=["accounts"])
bearer = HTTPBearer(auto_error=False)

class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    password: str = Field(min_length=8, max_length=128)


def password_hash(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 600_000).hex()
    return f"{salt}${digest}"


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer), db: Session = Depends(get_db)) -> User:
    if credentials is None:
        raise HTTPException(401, "Please log in", headers={"WWW-Authenticate": "Bearer"})
    token_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    session = db.get(AuthSession, token_hash)
    if session is None or session.expires_at <= datetime.utcnow():
        raise HTTPException(401, "Your session has expired. Please log in again.")
    user = db.get(User, session.user_id)
    if user is None:
        raise HTTPException(401, "Please log in")
    return user


def new_session(db: Session, user: User) -> dict:
    token = secrets.token_urlsafe(32)
    db.add(AuthSession(token_hash=hashlib.sha256(token.encode()).hexdigest(), user_id=user.id,
                       expires_at=datetime.utcnow() + timedelta(days=7)))
    db.commit()
    return {"token": token, "username": user.username}


@router.post("/register", status_code=201)
def register(data: Credentials, db: Session = Depends(get_db)):
    username = data.username.strip().casefold()
    if len(username) < 3:
        raise HTTPException(422, "Username must contain at least 3 characters")
    user = User(username=username, password_hash=password_hash(data.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "This username is already taken")
    db.refresh(user)
    return new_session(db, user)


@router.post("/login")
def login(data: Credentials, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username.strip().casefold()).first()
    stored = user.password_hash if user else password_hash("dummy-password")
    if not hmac.compare_digest(stored, password_hash(data.password, stored.split("$")[0])) or user is None:
        raise HTTPException(401, "Incorrect username or password")
    return new_session(db, user)


@router.post("/logout")
def logout(user: User = Depends(current_user), credentials: HTTPAuthorizationCredentials = Depends(bearer), db: Session = Depends(get_db)):
    session = db.get(AuthSession, hashlib.sha256(credentials.credentials.encode()).hexdigest())
    db.delete(session)
    db.commit()
    return {"success": True}
