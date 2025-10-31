#!/usr/bin/env python3
"""
Suno Audio Generation Module - Browser-Based Authentication Implementation

OVERVIEW
========
This module generates audio tracks via Suno's API by replicating the browser's authentication
and API interaction patterns. It requires valid session cookies from a logged-in Suno account
with available credits.

AUTHENTICATION ARCHITECTURE
============================
Suno uses Clerk (https://clerk.com) as their authentication provider. Understanding this is
critical for maintaining this code:

1. CLERK SESSION MANAGEMENT
   - Clerk maintains server-side sessions identified by session_id (e.g., "sess_XXX")
   - Sessions have TWO expiration mechanisms:
     a) Token expiration: ~1 hour (refreshable via /touch endpoint)
     b) Absolute session expiration: Hours/days (requires re-authentication in browser)
   
2. AUTHENTICATION COOKIES (stored in authentication.json)
   - __session: JWT containing session claims (sid, exp, iat, fva, user info)
     * This is the PRIMARY authentication token
     * Issued by Clerk, validated by Suno
     * Contains 'fva' (factor verification age) - minutes since last password entry
   
   - __client_uat: Unix timestamp of last client update
     * Updated by Clerk on session activity
     * Used for session synchronization
   
   - __client: JWT containing client metadata
     * Optional but recommended
     * Contains client_id and rotating_token
   
   - device_id: Persistent UUID for device fingerprinting
     * Generated once and stored in authentication.json
     * Used by Suno's anti-bot system to identify trusted devices

TOKEN REFRESH FLOW
===================
The browser automatically refreshes tokens to maintain long sessions. We replicate this:

1. STARTUP VALIDATION (in __init__)
   - Extract session_id from __session JWT
   - Call _touch_session() to validate and refresh
   - Updates authentication.json with fresh tokens
   - Updates in-memory Pydantic models

2. PRE-GENERATION REFRESH (in _mint_bearer)
   - Call Clerk's /touch endpoint with current cookies
   - Touch returns NEW __session JWT in response body: response.last_active_token.jwt
   - This new JWT has refreshed expiration (iat/exp moved forward)
   - Extract __client_uat from Set-Cookie headers
   - Atomically update authentication.json and in-memory state
   
3. WHY BOTH PLACES?
   - Startup: Catch expired sessions early, fail fast with clear error
   - Pre-generation: Ensure fresh token for each generation attempt
   - Mirrors browser behavior: touches on page load and before sensitive actions

ANTI-BOT MECHANISMS
===================
Suno employs multiple anti-bot checks that we must satisfy:

1. CAPTCHA CHECK (Critical - added to fix "Token validation failed")
   - Endpoint: POST /api/c/check with {"ctype": "generation"}
   - Response: {"required": false} = proceed, {"required": true} = captcha needed
   - Called BEFORE each generation attempt
   - High 'fva' (factor verification age) can trigger captcha requirement
   - Browser handles this automatically, we must call it explicitly
   
2. BROWSER-TOKEN HEADER
   - Format: {"token": "BASE64({\"timestamp\": MILLISECONDS})"}
   - Fresh timestamp generated for each API request
   - Anti-replay protection mechanism
   - Example: {"token": "eyJ0aW1lc3RhbXAiOjE3NjE5MjQxMTM0ODh9"}
   
3. DEVICE-ID HEADER
   - Persistent UUID stored in authentication.json
   - Identifies device across sessions
   - Trusted devices get lenient captcha treatment
   - Format: "14384c7f-2485-4c16-83ef-7ff507937460"

BEARER TOKEN vs SESSION COOKIE
===============================
IMPORTANT: These are the SAME JWT, used in different contexts:

1. As SESSION COOKIE (__session):
   - Sent to Clerk endpoints for authentication
   - Used in Cookie header
   - Refreshed via /touch endpoint

2. As BEARER TOKEN:
   - Sent to Suno API endpoints (studio-api.prod.suno.com)
   - Used in Authorization: Bearer {jwt} header
   - The JWT from /touch response IS the bearer token
   - Has 1-hour validity, contains 'aud: suno-api'

CLERK ENDPOINTS USED
====================
1. POST /v1/client/sessions/{session_id}/touch
   - Query params: __clerk_api_version=2025-04-10, _clerk_js_version=5.103.1
   - Form data: Same params + active_organization_id=""
   - Content-Type: application/x-www-form-urlencoded
   - Returns: Fresh JWT in response.last_active_token.jwt
   - Also returns: Set-Cookie headers with updated __client_uat

SUNO ENDPOINTS USED
===================
1. POST /api/c/check (CRITICAL - added to fix generation failures)
   - Headers: Authorization, browser-token, device-id
   - Body: {"ctype": "generation"}
   - Returns: {"required": false|true}
   - Must be called before EVERY generation

2. POST /api/generate/v2-web/
   - Headers: Authorization, browser-token, device-id, Content-Type, Origin, Referer
   - Body: Generation parameters (prompt, model, metadata)
   - Returns: Clip IDs and request ID

3. GET /api/feed/v2?ids={clip_ids}
   - Headers: Same anti-bot headers as generation
   - Polls for clip completion status

COMMON FAILURE MODES & FIXES
==============================
1. "Token validation failed" (422)
   - CAUSE: Missing /api/c/check call OR high fva triggered captcha
   - FIX: Ensure _check_captcha_required() called before generation
   - If captcha required=true: Get fresh cookies from browser

2. "Clerk session has expired on the server"
   - CAUSE: Absolute session expiration (not just token expiration)
   - FIX: User must log into suno.com in browser, copy fresh cookies
   - Touch endpoint cannot revive truly dead sessions

3. Missing anti-bot headers
   - CAUSE: browser-token or device-id not sent
   - FIX: Ensure both generated for ALL Suno API calls

4. Race conditions in authentication.json
   - CAUSE: Multiple processes writing simultaneously
   - FIX: Use tempfile + os.replace for atomic writes (already implemented)

DEBUGGING TIPS
==============
1. Check JWT expiration:
   - Decode __session JWT (base64 decode the middle part)
   - Check 'exp' claim against current Unix time
   - Check 'fva' claim - high values (>800 minutes) may trigger captcha

2. Compare with browser:
   - Open DevTools → Network tab
   - Watch requests to clerk.suno.com and studio-api.prod.suno.com
   - Compare headers and payloads with our implementation
   - Look for any new headers or endpoints being called

3. Enable debug logging:
   - Set logger to DEBUG level
   - Check: "Captcha check response", "Bearer token validated", "Session touched"
   - Verify authentication.json is being updated with fresh timestamps

AUTHENTICATION.JSON STRUCTURE
==============================
{
  "suno": {
    "session": "eyJhbGc...",      // JWT - refreshed automatically
    "client_uat": "1761870513",   // timestamp - refreshed automatically  
    "client": "eyJhbGc...",       // JWT - static, rarely changes
    "device_id": "uuid-v4"        // UUID - generated once, persistent
  }
}

MAINTENANCE NOTES
=================
- Clerk API version (2025-04-10) and JS version (5.103.1) may need updates
- Monitor browser for new anti-bot headers or endpoint changes
- If Suno changes auth providers, this entire module needs rewrite
- /api/c/check is critical - if removed, generation will fail
- Factor verification age (fva) thresholds may change over time
"""


from __future__ import annotations

import base64
import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import requests

from pipeline_utilities.args import parse_arguments
from pipeline_utilities.configuration import load_configuration
from pipeline_utilities.logs import EnhancedLogger
from pipeline_utilities.paths import Paths, Project
from pipeline_utilities.authentication import load_authentication
from pipeline_utilities.network import NetworkRetry


class Config:
    """ Configuration constants for Suno API interaction """
    CLERK_BASE: Final[str] = "https://clerk.suno.com"
    API_BASE: Final[str] = "https://studio-api.prod.suno.com"
    SITE_ORIGIN: Final[str] = "https://suno.com"
    POLL_INTERVAL_SECONDS: Final[int] = 10
    TIMEOUT_SECONDS: Final[int] = 600
    PASS_DELAY_SECONDS: Final[int] = 10
    DEFAULT_HEADERS_JSON: Final[Dict[str, str]] = {
        "Content-Type": "application/json",
        "Origin": "https://suno.com",
        "Referer": "https://suno.com/",
        "Accept": "application/json",
    }
    DEFAULT_HEADERS_GET_JSON: Final[Dict[str, str]] = {
        "Origin": "https://suno.com",
        "Referer": "https://suno.com/",
        "Accept": "application/json",
    }


class SunoAudioGenerator:
    """ Orchestrates Suno generations with pass -based saving """

    def __init__(self, logger: EnhancedLogger, auth_data: Any):
        self.logger = logger
        self.auth_data = auth_data

        # Validate session at startup
        session_cookie, _, _ = self._get_cookies()
        try:
            parts = session_cookie.split(".")
            if len(parts) >= 2:
                payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
                claims = json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
                session_id = claims.get("sid")
                if session_id and isinstance(session_id, str) and session_id.startswith("sess_"):
                    self._touch_session(session_id)
                    self.logger.info("Session validated and refreshed at startup")
        except Exception as exc:
            self.logger.warning(f"Session touch at startup failed: {exc}, will retry during token mint")

    def _mint_bearer(self, session_cookie: str, client_uat_cookie: str, client_cookie: Optional[str]) -> str:
        """ Mint a short - lived Clerk session token using provided cookies """
        cookies = {
            "__session": session_cookie,
            "__client_uat": client_uat_cookie,
        }
        if client_cookie:
            cookies["__client"] = client_cookie

        s = requests.Session()
        s.headers.update({
            "Origin": Config.SITE_ORIGIN,
            "Referer": f"{Config.SITE_ORIGIN}/",
            "Accept": "application/json",
        })
        s.cookies.update(cookies)

        self.logger.debug("=" * 80)
        self.logger.debug("MINTING BEARER TOKEN - Step 1: Extract session ID from JWT")
        self.logger.debug("=" * 80)

        # Extract session_id from JWT claims
        # Try __client cookie first (most likely to contain sid), then __session
        session_id: Optional[str] = None
        cookies_to_try = []
        if client_cookie:
            cookies_to_try.append(("__client", client_cookie))
        cookies_to_try.append(("__session", session_cookie))

        self.logger.debug("Attempting to extract session_id from JWT")
        for cookie_name, cookie_value in cookies_to_try:
            try:
                self.logger.debug("Trying to extract sid from %s cookie", cookie_name)
                parts = cookie_value.split(".")
                self.logger.debug("JWT parts count: %s", len(parts))
                if len(parts) >= 2:
                    payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
                    claims = json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
                    self.logger.debug("JWT claims extracted from %s: %s", cookie_name, json.dumps(claims, indent=2))
                    sid = claims.get("sid")
                    self.logger.debug("Extracted sid from %s JWT: %s", cookie_name, sid)
                    if isinstance(sid, str) and sid.startswith("sess_"):
                        session_id = sid
                        self.logger.debug("Successfully extracted session_id from %s cookie: %s",
                                          cookie_name, session_id)
                        break
                    else:
                        self.logger.debug("sid from %s JWT does not start with 'sess_': %s", cookie_name, sid)
                else:
                    self.logger.debug("%s JWT does not have enough parts (need at least 2, got %s)",
                                      cookie_name, len(parts))
            except (ValueError, json.JSONDecodeError, IndexError) as exc:
                self.logger.debug("Failed to parse %s cookie JWT: %s", cookie_name, exc)
                continue

        if not session_id:
            self.logger.error("Failed to extract session_id from JWT")
            raise RuntimeError("No active Clerk session id available")

        # Touch session to get fresh bearer token (matches browser behavior)
        # The touch response contains last_active_token.jwt which is a full 1-hour bearer token
        self.logger.debug("Touching session to get bearer token")

        # Match browser request exactly: query params + form data
        touch_url = f"{Config.CLERK_BASE}/v1/client/sessions/{session_id}/touch"
        params = {
            "__clerk_api_version": "2025-04-10",
            "_clerk_js_version": "5.103.1"
        }
        form_data = {
            "__clerk_api_version": "2025-04-10",
            "_clerk_js_version": "5.103.1",
            "active_organization_id": ""
        }

        self.logger.debug(f"POST URL: {touch_url} (with query params)")

        r = NetworkRetry.execute(lambda: s.post(
            touch_url,
            params=params,
            data=form_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15
        ))

        # Extract new session JWT and cookies from response
        try:
            response_data = r.json()
            if response_data.get("response"):
                session_data = response_data["response"]
            else:
                session_data = response_data

            # Extract the bearer token from last_active_token.jwt
            last_active_token = session_data.get("last_active_token", {})
            bearer_token = last_active_token.get("jwt")

            if not bearer_token:
                self.logger.error("Touch response missing last_active_token.jwt")
                raise RuntimeError("Failed to get bearer token from touch response")

            # Validate the bearer token is not expired
            try:
                parts = bearer_token.split(".")
                if len(parts) >= 2:
                    payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
                    claims = json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
                    exp = claims.get("exp", 0)
                    now = int(time.time())

                    if exp <= now:
                        self.logger.error("Touch endpoint returned EXPIRED token! exp=%s, now=%s, expired %s seconds ago",
                                          exp, now, now - exp)
                        raise RuntimeError(
                            "Clerk session has expired on the server. Please refresh your session cookies from the browser:\n"
                            "1. Go to suno.com and ensure you're logged in\n"
                            "2. Open DevTools > Network > Find a request to clerk.suno.com\n"
                            "3. Copy the __session, __client_uat, and __client cookies\n"
                            "4. Update authentication.json with fresh values"
                        )

                    time_remaining = exp - now
                    self.logger.debug(
                        f"Bearer token validated: expires in {time_remaining} seconds ({time_remaining/3600:.1f} hours)")
            except (ValueError, json.JSONDecodeError, KeyError, IndexError) as parse_err:
                self.logger.warning("Could not validate bearer token expiration: %s", parse_err)

            self.logger.debug(f"Bearer token extracted from touch response (length: {len(bearer_token)})")

            # Also update stored session cookies for next time
            new_session_jwt = bearer_token  # The bearer token IS the new session JWT
            new_cookies = self._extract_cookies_from_response(r)
            new_cookies["__session"] = new_session_jwt

            if new_cookies:
                self._update_stored_cookies(new_cookies)
                self.logger.debug("Updated stored cookies from touch response")

            self.logger.debug("=" * 80)
            return bearer_token
        except (ValueError, json.JSONDecodeError, KeyError) as exc:
            self.logger.error("Failed to extract bearer token from touch response: %s", exc)
            raise RuntimeError("Failed to get bearer token from touch response") from exc

    def _check_captcha_required(self, bearer: str) -> bool:
        """ Check if captcha is required before generation. Returns True if captcha needed """
        timestamp_ms = int(time.time() * 1000)
        timestamp_payload = json.dumps({"timestamp": timestamp_ms})
        browser_token_jwt = base64.urlsafe_b64encode(timestamp_payload.encode()).decode().rstrip('=')
        browser_token = json.dumps({"token": browser_token_jwt})
        device_id = self._get_device_id()

        headers = {
            **Config.DEFAULT_HEADERS_JSON,
            "Authorization": f"Bearer {bearer}",
            "browser-token": browser_token,
            "device-id": device_id,
        }

        url = f"{Config.API_BASE}/api/c/check"
        body = {"ctype": "generation"}

        self.logger.debug("Checking if captcha is required for generation")
        try:
            r = NetworkRetry.execute(
                lambda: requests.post(url, json=body, headers=headers, timeout=15)
            )
            data = r.json()
            required = data.get("required", False)
            self.logger.debug(f"Captcha check response: {data}")
            return required
        except Exception as exc:
            self.logger.warning(f"Captcha check failed: {exc}, assuming not required")
            return False

    def _start_generation(self, bearer: str, prompt: str) -> Tuple[str, List[str]]:
        """ Submit a generation request. Returns(request_id, clip_ids) """
        # Check if captcha is required (Suno's anti-bot check)
        captcha_required = self._check_captcha_required(bearer)
        if captcha_required:
            raise RuntimeError(
                "Captcha is required for generation. This typically happens when:\n"
                "1. The session hasn't been used for authentication in a while\n"
                "2. Suno's anti-bot system flags the request\n"
                "Please go to suno.com in your browser, complete any captcha if shown, "
                "and then copy fresh session cookies to authentication.json"
            )

        create_session_token = str(uuid.uuid4())
        transaction_uuid = str(uuid.uuid4())

        body = {
            "generation_type": "TEXT",
            "mv": "chirp-auk",
            "prompt": "",
            "gpt_description_prompt": prompt,
            "make_instrumental": True,
            "user_uploaded_images_b64": None,
            "metadata": {
                "web_client_pathname": "/create",
                "is_max_mode": False,
                "create_mode": "simple",
                "disable_volume_normalization": False,
                "can_control_sliders": [],
                "lyrics_model": "default",
                "create_session_token": create_session_token,
            },
            "override_fields": [],
            "cover_clip_id": None,
            "cover_start_s": None,
            "cover_end_s": None,
            "persona_id": None,
            "artist_clip_id": None,
            "artist_start_s": None,
            "artist_end_s": None,
            "continue_clip_id": None,
            "continued_aligned_prompt": None,
            "continue_at": None,
            "transaction_uuid": transaction_uuid,
        }

        # Generate browser-token (timestamp-based JWT for anti-replay)
        timestamp_ms = int(time.time() * 1000)
        timestamp_payload = json.dumps({"timestamp": timestamp_ms})
        browser_token_jwt = base64.urlsafe_b64encode(timestamp_payload.encode()).decode().rstrip('=')
        browser_token = json.dumps({"token": browser_token_jwt})

        # Get or generate device-id (persistent UUID for device fingerprinting)
        device_id = self._get_device_id()

        headers = {
            **Config.DEFAULT_HEADERS_JSON,
            "Authorization": f"Bearer {bearer}",
            "browser-token": browser_token,
            "device-id": device_id,
        }
        url = f"{Config.API_BASE}/api/generate/v2-web/"
        body_json = json.dumps(body)

        self.logger.debug("Bearer token being used (length: %s, first 50 chars: %s...)",
                          len(bearer), bearer[:50] if len(bearer) > 50 else bearer)

        self.logger.debug("=" * 80)
        self.logger.debug("STARTING GENERATION REQUEST")
        self.logger.debug("=" * 80)
        self.logger.debug("URL: %s", url)
        self.logger.debug("Method: POST")
        self.logger.debug("Headers: %s", json.dumps(headers, indent=2))
        self.logger.debug("Request Body: %s", body_json)
        self.logger.debug("-" * 80)

        try:
            r = NetworkRetry.execute(
                lambda: requests.post(
                    url,
                    headers=headers,
                    data=body_json,
                    timeout=30
                )
            )
            self.logger.debug("Response Status: %s", r.status_code)
            self.logger.debug("Response Headers: %s", json.dumps(dict(r.headers), indent=2))
            resp = r.json()
            self.logger.debug("Response Body: %s", json.dumps(resp, indent=2))
            self.logger.debug("=" * 80)
        except requests.exceptions.HTTPError as exc:
            self.logger.error("=" * 80)
            self.logger.error("HTTP ERROR IN GENERATION REQUEST")
            self.logger.error("=" * 80)
            self.logger.error("URL: %s", url)
            self.logger.error("Method: POST")
            self.logger.error("Status Code: %s", exc.response.status_code if exc.response else 'N/A')
            if exc.response:
                self.logger.error("Response Headers: %s", json.dumps(dict(exc.response.headers), indent=2))
                try:
                    error_body = exc.response.json()
                    self.logger.error("Error Response Body: %s", json.dumps(error_body, indent=2))
                except (ValueError, json.JSONDecodeError):
                    self.logger.error("Error Response Body (raw): %s", exc.response.text)
            self.logger.error("Request Body Sent: %s", body_json)
            self.logger.error("=" * 80)
            raise
        clips = resp.get("clips") or []
        clip_ids = [c.get("id") for c in clips if isinstance(c, dict) and c.get("id")]
        if not clip_ids:
            raise RuntimeError("Generate returned no clips")
        request_id = resp.get("id") or ""
        return request_id, clip_ids

    def _poll_feed(self, bearer: str, clip_ids: List[str]) -> Dict[str, dict]:
        """ Poll feed for provided clip IDs; returns mapping id -> clip object """
        ids_param = ",".join(clip_ids)

        # Generate browser-token and device-id for API validation
        timestamp_ms = int(time.time() * 1000)
        timestamp_payload = json.dumps({"timestamp": timestamp_ms})
        browser_token_jwt = base64.urlsafe_b64encode(timestamp_payload.encode()).decode().rstrip('=')
        browser_token = json.dumps({"token": browser_token_jwt})
        device_id = self._get_device_id()

        headers = {
            **Config.DEFAULT_HEADERS_GET_JSON,
            "Authorization": f"Bearer {bearer}",
            "browser-token": browser_token,
            "device-id": device_id,
        }
        r = requests.get(f"{Config.API_BASE}/api/feed/v2", params={"ids": ids_param}, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        out: Dict[str, dict] = {}
        for clip in (data.get("clips") or []):
            if isinstance(clip, dict) and clip.get("id"):
                out[clip["id"]] = clip
        return out

    def _download_clip_to(self, dest_path: Path, audio_url: str) -> None:
        """ Download the clip audio to the destination path """
        r = NetworkRetry.execute(lambda: requests.get(audio_url, stream=True, timeout=60))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with r:
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    def _save_clip_json(self, dest_path: Path, clip_obj: dict) -> None:
        """ Save clip JSON to the destination path """
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(clip_obj, f, indent=2, ensure_ascii=False)

    def _get_cookies(self) -> Tuple[str, str, Optional[str]]:
        """ Extract required cookies from authentication data """
        if not self.auth_data:
            raise ValueError("authentication.json missing 'suno' object with cookies")

        if not self.auth_data.suno:
            raise ValueError("authentication.json missing 'suno' object with cookies")

        suno = self.auth_data.suno
        session_cookie = suno.session
        client_uat_cookie = suno.client_uat
        client_cookie = suno.client

        if not session_cookie or not client_uat_cookie:
            raise ValueError("Missing required Suno cookies: 'session' and 'client_uat'")
        return session_cookie, client_uat_cookie, client_cookie

    def _get_device_id(self) -> str:
        """ Get or generate persistent device ID for browser fingerprinting """
        auth_path = Path(Project.get_root_path("authentication.json"))

        # Try to read existing device_id from authentication.json
        try:
            with open(auth_path, 'r', encoding='utf-8') as f:
                auth_data = json.load(f)

            if 'suno' in auth_data and 'device_id' in auth_data['suno']:
                return auth_data['suno']['device_id']
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

        # Generate new device_id (UUID4)
        device_id = str(uuid.uuid4())
        self.logger.debug("Generated new device_id: %s", device_id)

        # Store it in authentication.json
        try:
            with open(auth_path, 'r', encoding='utf-8') as f:
                auth_data = json.load(f)

            if 'suno' not in auth_data:
                auth_data['suno'] = {}

            auth_data['suno']['device_id'] = device_id

            # Write atomically
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=auth_path.parent,
                delete=False,
                suffix='.tmp'
            )
            try:
                json.dump(auth_data, temp_file, indent=2, ensure_ascii=False)
                temp_file.close()
                Path(temp_file.name).replace(auth_path)
                self.logger.debug("Stored device_id in authentication.json")
            except Exception as exc:
                self.logger.warning(f"Failed to store device_id: {exc}")
                Path(temp_file.name).unlink(missing_ok=True)
        except Exception as exc:
            self.logger.warning(f"Failed to store device_id: {exc}")

        return device_id

    def _extract_cookies_from_response(self, response: requests.Response) -> Dict[str, str]:
        """ Extract Clerk cookies from response Set - Cookie headers """
        cookies = {}

        # Get all Set-Cookie headers
        set_cookie_headers = response.headers.get_list('Set-Cookie') if hasattr(response.headers, 'get_list') else []

        # Fallback for requests library which doesn't have get_list
        if not set_cookie_headers and 'Set-Cookie' in response.headers:
            # requests merges multiple headers, try to get from raw response
            if hasattr(response.raw, '_original_response'):
                set_cookie_headers = response.raw._original_response.msg.get_all('Set-Cookie') or []

        for cookie_header in set_cookie_headers:
            # Parse cookie: "name=value; Path=/; HttpOnly; ..."
            if not cookie_header:
                continue

            # Split on first semicolon to separate name=value from attributes
            parts = cookie_header.split(';', 1)
            if not parts:
                continue

            name_value = parts[0].strip()
            if '=' not in name_value:
                continue

            name, value = name_value.split('=', 1)
            name = name.strip()
            value = value.strip()

            # Only extract the cookies we care about
            if name == '__session':
                cookies['__session'] = value
                self.logger.debug(f"Extracted __session cookie from response (length: {len(value)})")
            elif name == '__client_uat':
                cookies['__client_uat'] = value
                self.logger.debug(f"Extracted __client_uat cookie from response: {value}")
            elif name == '__client':
                cookies['__client'] = value
                self.logger.debug(f"Extracted __client cookie from response (length: {len(value)})")

        return cookies

    def _update_stored_cookies(self, new_cookies: Dict[str, str]) -> None:
        """ Update cookies in both authentication.json file and in -memory Pydantic model """
        if not new_cookies:
            return

        auth_path = Path(Project.get_root_path("authentication.json"))

        # Read current authentication.json
        with open(auth_path, 'r', encoding='utf-8') as f:
            auth_json = json.load(f)

        # Update the suno cookies if present in new_cookies
        if 'suno' not in auth_json:
            self.logger.warning("No 'suno' section in authentication.json, cannot update cookies")
            return

        updated_cookies = []

        if '__session' in new_cookies:
            auth_json['suno']['session'] = new_cookies['__session']
            self.auth_data.suno.session = new_cookies['__session']
            updated_cookies.append('session')
            self.logger.debug("Updated __session cookie in file and memory")

        if '__client_uat' in new_cookies:
            auth_json['suno']['client_uat'] = new_cookies['__client_uat']
            self.auth_data.suno.client_uat = new_cookies['__client_uat']
            updated_cookies.append('client_uat')
            self.logger.debug("Updated __client_uat cookie in file and memory")

        if '__client' in new_cookies:
            auth_json['suno']['client'] = new_cookies['__client']
            self.auth_data.suno.client = new_cookies['__client']
            updated_cookies.append('client')
            self.logger.debug("Updated __client cookie in file and memory")

        if not updated_cookies:
            self.logger.debug("No recognized cookies to update")
            return

        # Write atomically: create temp file, write, then rename
        auth_dir = auth_path.parent
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=auth_dir,
            delete=False,
            suffix='.json'
        ) as tmp_file:
            json.dump(auth_json, tmp_file, indent=2)
            tmp_path = Path(tmp_file.name)

        # Atomic rename
        tmp_path.replace(auth_path)

        self.logger.debug(f"Persisted updated cookies to authentication.json: {', '.join(updated_cookies)}")

    def _touch_session(self, session_id: str) -> None:
        """ Touch session to refresh cookies(matches browser behavior) """
        session_cookie, client_uat_cookie, client_cookie = self._get_cookies()

        # Create session with cookies
        cookies = {
            "__session": session_cookie,
            "__client_uat": client_uat_cookie,
        }
        if client_cookie:
            cookies["__client"] = client_cookie

        s = requests.Session()
        s.headers.update({
            "Origin": Config.SITE_ORIGIN,
            "Referer": f"{Config.SITE_ORIGIN}/",
            "Accept": "application/json",
        })
        s.cookies.update(cookies)

        # Touch the session (match browser request exactly)
        touch_url = f"{Config.CLERK_BASE}/v1/client/sessions/{session_id}/touch"
        params = {
            "__clerk_api_version": "2025-04-10",
            "_clerk_js_version": "5.103.1"
        }
        form_data = {
            "__clerk_api_version": "2025-04-10",
            "_clerk_js_version": "5.103.1",
            "active_organization_id": ""
        }

        self.logger.debug(f"Touching session: {touch_url}")

        r = NetworkRetry.execute(lambda: s.post(
            touch_url,
            params=params,
            data=form_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15
        ))

        # Extract cookies from Set-Cookie headers (for __client_uat)
        new_cookies = self._extract_cookies_from_response(r)

        # Extract new session token from JSON response body (last_active_token.jwt)
        try:
            response_data = r.json()
            if response_data.get("response"):
                # Touch returns nested structure with "response" key
                session_data = response_data["response"]
            else:
                session_data = response_data

            # Get the fresh JWT from last_active_token
            last_active_token = session_data.get("last_active_token", {})
            new_session_jwt = last_active_token.get("jwt")

            if new_session_jwt:
                new_cookies["__session"] = new_session_jwt
                self.logger.debug(f"Extracted new session JWT from response body (length: {len(new_session_jwt)})")
            else:
                self.logger.warning("Touch response did not contain last_active_token.jwt")
        except (ValueError, json.JSONDecodeError, KeyError) as exc:
            self.logger.warning(f"Failed to parse touch response for session JWT: {exc}")

        if new_cookies:
            self._update_stored_cookies(new_cookies)
            self.logger.info("Session touched, cookies refreshed")
        else:
            self.logger.debug("Session touched, but no new cookies in response")

    def _ensure_prompt(self, step_config: Any) -> str:
        """ Get the first prompt from the step configuration """
        prompts = getattr(step_config, "prompts", None)
        if not prompts or len(prompts) == 0 or not prompts[0] or prompts[0].strip() == "":
            raise ValueError("Step is missing prompts[0] for Suno generation")
        return prompts[0]

    def _save_completed_clips(self, clips_state: Dict[str, dict], stem: str, suffix: str,
                              start_index: int) -> int:
        """ Save all completed clips to interim path with sequential pass indices

        Returns the next index after saving
        """
        clip_ids_sorted = sorted(clips_state.keys())
        index = start_index
        for cid in clip_ids_sorted:
            clip = clips_state[cid]
            status = clip.get("status")
            if status != "complete":
                continue

            audio_url = clip.get("audio_url")
            if not audio_url:
                raise RuntimeError(f"Clip {cid} complete but missing audio_url")

            mp3_filename = f"{stem}_pass_{index:03d}{suffix}"
            json_filename = f"{stem}_pass_{index:03d}.json"

            mp3_path = Paths.get_interim_path(mp3_filename)
            json_path = Paths.get_interim_path(json_filename)

            self.logger.info(f"Saving clip {cid} -> {mp3_filename}")
            self.logger.debug(f"Audio URL: {audio_url}")

            self._save_clip_json(json_path, clip)
            self._download_clip_to(mp3_path, audio_url)
            index += 1

        return index

    def execute_single_generation(self, prompt: str) -> List[str]:
        """ Execute a single generation cycle and return the resulting clip ids """
        session_cookie, client_uat_cookie, client_cookie = self._get_cookies()
        bearer = self._mint_bearer(session_cookie, client_uat_cookie, client_cookie)
        _, clip_ids = self._start_generation(bearer, prompt)

        # Poll until all clip ids complete or timeout
        pending = set(clip_ids)
        last_state: Dict[str, dict] = {}
        deadline = time.time() + Config.TIMEOUT_SECONDS

        while pending and time.time() < deadline:
            time.sleep(Config.POLL_INTERVAL_SECONDS)
            bearer = self._mint_bearer(session_cookie, client_uat_cookie, client_cookie)
            state = self._poll_feed(bearer, sorted(pending))

            for cid in sorted(pending):
                clip = state.get(cid)
                status = clip.get("status") if clip else "unknown"
                self.logger.info(f"  * {cid} - {status}")

            for cid, clip in state.items():
                last_state[cid] = clip
                if clip.get("status") == "complete" and cid in pending:
                    pending.remove(cid)

        if pending:
            raise TimeoutError(f"Timed out waiting for clips: {', '.join(sorted(pending))}")

        # Return completed clips in the order we saw them
        return sorted(last_state.keys())

    def execute_step(self, step_name: str, config_data: Any) -> None:
        """ Execute the configured Suno generation step """
        start_time = time.time()
        self.logger.header(f"Executing step: {step_name}")

        step_config = config_data.steps.get(step_name)
        if not step_config:
            raise ValueError(f"Step '{step_name}' not found in configuration")

        prompt = self._ensure_prompt(step_config)

        # Derive naming from output
        output_path = Path(step_config.output)
        stem = output_path.stem
        suffix = output_path.suffix

        self.logger.info(f"Step: {step_name}")
        self.logger.info(f"Module: {step_config.module}")
        self.logger.info(f"Output base: {step_config.output}")
        self.logger.info("Saving all clips with sequential pass numbering")

        next_index = 1
        total_generations = step_config.passes if step_config.passes is not None else 1
        self.logger.info(f"Generations to run: {total_generations}")

        for gen_num in range(1, total_generations + 1):
            self.logger.info(f"Generation {gen_num}/{total_generations}")
            # Execute generation and poll
            clip_ids = self.execute_single_generation(prompt)

            # After completion, get final state once to save JSON accurately
            session_cookie, client_uat_cookie, client_cookie = self._get_cookies()
            bearer = self._mint_bearer(session_cookie, client_uat_cookie, client_cookie)
            final_state = self._poll_feed(bearer, clip_ids)

            # Save completed clips and advance index
            next_index = self._save_completed_clips(final_state, stem, suffix, next_index)

            # Delay before next pass to avoid rate limiting
            if gen_num < total_generations:
                self.logger.info(f"Waiting {Config.PASS_DELAY_SECONDS} seconds before next generation")
                time.sleep(Config.PASS_DELAY_SECONDS)

        duration = time.time() - start_time
        self.logger.header("Step completed successfully")
        self.logger.info(f"Step name: {step_name}")
        self.logger.info(f"Module: {step_config.module}")
        self.logger.info(f"Total outputs saved: {next_index - 1}")
        self.logger.info(f"Duration: {duration:.2f} seconds")


def main() -> None:
    """ Main entry point """
    args = parse_arguments("Suno audio generation module")

    # Load configuration first (for debug flag)
    config_path = Project.get_configuration()
    config_loader = load_configuration(config_path)
    config_data = config_loader.data

    # Setup logging
    logger = EnhancedLogger.setup_pipeline_logging(
        log_file=args.log_file,
        debug=config_data.debug or False,
        script_name="suno_audio"
    )

    logger.info(f"Loaded configuration: {config_data.name}")

    # Load authentication
    try:
        auth_path = Project.get_root_path("authentication.json")
        auth_loader = load_authentication(auth_path)
        auth_data = auth_loader.data
        logger.info("Loaded authentication data")
    except FileNotFoundError as exc:
        raise RuntimeError("authentication.json file not found") from exc
    except ValueError as exc:
        raise RuntimeError(f"Failed to load authentication data: {exc}") from exc

    try:
        generator = SunoAudioGenerator(logger, auth_data)
        generator.execute_step(args.step, config_data)
    except Exception as exc:
        logger.error(f"Suno audio generation failed: {exc}")
        raise


if __name__ == "__main__":
    main()
