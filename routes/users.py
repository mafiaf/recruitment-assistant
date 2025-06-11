from fastapi import APIRouter, Request, Form, Depends, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
import os
import main

router = APIRouter()

PHOTO_DIR = os.path.join("static", "photos")
os.makedirs(PHOTO_DIR, exist_ok=True)

@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return await main.render(request, "login.html", page_title="Login", active="/login")

@router.post("/login")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    user = await main.verify_password(username, password)
    if not user:
        return await main.render(
            request,
            "login.html",
            {"error": "Invalid credentials"},
            page_title="Login",
            active="/login",
            status_code=401,
        )
    resp = RedirectResponse("/", status_code=303)
    main.set_session(resp, user)
    return resp

@router.get("/logout")
def logout():
    resp = RedirectResponse("/", status_code=303)
    main.clear_session(resp)
    return resp

@router.get("/admin/users", response_class=HTMLResponse, dependencies=[Depends(main.require_owner)])
async def user_admin(request: Request):
    cursor = main.users_coll.find({}, {"_id": 0, "hashed_password": 0})
    users = await cursor.to_list(None)
    return await main.render(
        request,
        "admin_users.html",
        {"users": users},
        page_title="User admin",
        active="/admin/users",
    )

@router.post("/admin/users", dependencies=[Depends(main.require_owner)])
async def create_user_admin(
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form("user"),
):
    if await main.users_coll.find_one({"username": username}):
        raise HTTPException(400, "User already exists")
    await main.create_user(username, password, role)
    return RedirectResponse("/admin/users", status_code=303)

@router.post("/admin/users/delete", dependencies=[Depends(main.require_owner)])
async def delete_user_admin(username: str = Form(...), current=Depends(main.require_owner)):
    if username == current["username"]:
        raise HTTPException(400, "You cannot delete your own owner account")
    res = await main.users_coll.delete_one({"username": username})
    if res.deleted_count == 0:
        raise HTTPException(404, "User not found")
    return RedirectResponse("/admin/users", status_code=303)

@router.get("/admin/users/{username}/edit", response_class=HTMLResponse, dependencies=[Depends(main.require_owner)])
async def edit_user_form(request: Request, username: str):
    user = await main.users_coll.find_one({"username": username}, {"_id": 0, "hashed_password": 0})
    if not user:
        raise HTTPException(404, "User not found")
    return await main.render(
        request,
        "edit_user.html",
        {"user": user},
        page_title="Edit user",
        active="/admin/users",
    )

@router.post("/admin/users/{username}/edit", dependencies=[Depends(main.require_owner)])
async def update_user_admin(username: str, new_username: str = Form(...)):
    if username != new_username:
        if await main.users_coll.find_one({"username": new_username}):
            raise HTTPException(400, "Username already exists")
        res = await main.users_coll.update_one({"username": username}, {"$set": {"username": new_username}})
        if res.matched_count == 0:
            raise HTTPException(404, "User not found")
    return RedirectResponse("/admin/users", status_code=303)

@router.post("/admin/users/{username}/reset", dependencies=[Depends(main.require_owner)])
async def reset_user_password(username: str):
    res = await main.users_coll.update_one(
        {"username": username},
        {"$set": {"hashed_password": main.pwd_context.hash(main.DEFAULT_PASSWORD)}},
    )
    if res.matched_count == 0:
        raise HTTPException(404, "User not found")
    return RedirectResponse("/admin/users", status_code=303)

@router.get("/profile", response_class=HTMLResponse, dependencies=[Depends(main.require_login)])
async def profile(request: Request, user=Depends(main.get_current_user)):
    return await main.render(
        request,
        "profile.html",
        {"user": user},
        page_title="Profile",
        active="/profile",
    )

@router.post("/profile/password", dependencies=[Depends(main.require_login)])
async def change_password(old: str = Form(...), new: str = Form(...), user=Depends(main.get_current_user)):
    if not await main.verify_password(user["username"], old):
        raise HTTPException(400, "Old password incorrect")
    await main.users_coll.update_one(
        {"username": user["username"]},
        {"$set": {"hashed_password": main.pwd_context.hash(new)}},
    )
    return RedirectResponse("/profile?ok=1", status_code=303)


@router.post("/profile/photo", dependencies=[Depends(main.require_login)])
async def upload_photo(photo: UploadFile = File(...), user=Depends(main.get_current_user)):
    ext = os.path.splitext(photo.filename or "")[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".gif"}:
        raise HTTPException(400, "Invalid image type")
    filename = f"{user['username']}{ext}"
    path = os.path.join(PHOTO_DIR, filename)
    data = await photo.read()
    with open(path, "wb") as f:
        f.write(data)
    await main.users_coll.update_one(
        {"username": user["username"]},
        {"$set": {"photo": filename}},
    )
    return RedirectResponse("/profile", status_code=303)
