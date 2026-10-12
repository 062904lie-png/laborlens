import { getApp, getApps, initializeApp } from "https://www.gstatic.com/firebasejs/12.19.0/firebase-app.js";
import {
  getAuth,
  GoogleAuthProvider,
  getRedirectResult,
  signInWithPopup,
} from "https://www.gstatic.com/firebasejs/12.19.0/firebase-auth.js";

const firebaseConfig = Object.freeze({
  apiKey: "AIzaSyAcBscu2Fnidc_o9tUUv6EZcS1pE6vKJLA",
  authDomain: "laborlens-31a9c.firebaseapp.com",
  projectId: "laborlens-31a9c",
  appId: "1:373394325577:web:5aafdac3b9d114e6c8bf98",
});

let auth = null;
let provider = null;
let initializationError = null;

function errorTargetForCurrentForm() {
  const registerForm = document.getElementById("login-form")?.style.display === "none";
  return document.getElementById(registerForm ? "reg-err" : "login-err");
}

function showError(target, error) {
  const code = String(error?.code || "");
  const messages = {
    "auth/unauthorized-domain": "This website is not authorized for Firebase sign-in. Add laborlens-lac.vercel.app under Firebase Authentication → Settings → Authorized domains.",
    "auth/operation-not-allowed": "Google sign-in is disabled. Enable Google under Firebase Authentication → Sign-in method.",
    "auth/popup-blocked": "Your browser blocked the Google sign-in window. Open LaborLens directly in Safari or Chrome and allow popups for this site.",
    "auth/popup-closed-by-user": "The Google sign-in window was closed before sign-in finished.",
    "auth/cancelled-popup-request": "Another Google sign-in window is already open. Finish or close it, then try again.",
    "auth/api-key-not-valid.-please-pass-a-valid-api-key.": "Firebase rejected this web API key. Replace it with the current Web API key from Firebase Project settings → General, and check the key’s API restrictions.",
    "auth/invalid-api-key": "Firebase rejected this web API key. Replace it with the current Web API key from Firebase Project settings → General, and check the key’s API restrictions.",
  };
  if (target) {
    target.textContent = messages[code] || error?.message || "Google sign-in failed. Please try again.";
  }
}

function setBusy(busy) {
  document.querySelectorAll(".google-btn").forEach((button) => {
    button.disabled = busy;
    button.setAttribute("aria-busy", String(busy));
  });
}

async function exchangeFirebaseCredential(user, errorTarget) {
  if (typeof window.saveAuth !== "function") {
    throw new Error("LaborLens could not finish loading. Refresh the page and try again.");
  }
  const apiBase = String(window.LABORLENS_API_URL || "").replace(/\/+$/, "");
  if (!apiBase) throw new Error("LaborLens API address is missing from this page.");

  const idToken = await user.getIdToken();
  const response = await fetch(`${apiBase}/auth/firebase`, {
    method: "POST",
    cache: "no-store",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id_token: idToken }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "LaborLens could not verify your Google sign-in. Please try again.");
  }
  if (errorTarget) errorTarget.textContent = "";
  window.saveAuth(data);
}

try {
  const app = getApps().length ? getApp() : initializeApp(firebaseConfig);
  auth = getAuth(app);
  provider = new GoogleAuthProvider();
  provider.setCustomParameters({ prompt: "select_account" });
} catch (error) {
  initializationError = error;
}

window.llFirebaseGoogleSignIn = async (errorTarget = errorTargetForCurrentForm()) => {
  if (errorTarget) errorTarget.textContent = "";
  if (initializationError) {
    showError(errorTarget, initializationError);
    return;
  }
  if (!auth || !provider) {
    showError(errorTarget, new Error("Firebase sign-in did not initialize. Refresh the page and try again."));
    return;
  }

  setBusy(true);
  try {
    // Use popup auth on phones too. Redirect auth relies on third-party browser storage
    // when this Vercel site uses the Firebase-hosted authDomain, which modern mobile
    // browsers block. The popup flow avoids that storage dependency.
    const result = await signInWithPopup(auth, provider);
    await exchangeFirebaseCredential(result.user, errorTarget);
  } catch (error) {
    showError(errorTarget, error);
  } finally {
    setBusy(false);
  }
};

if (auth) {
  getRedirectResult(auth)
    .then((result) => result && exchangeFirebaseCredential(result.user, errorTargetForCurrentForm()))
    .catch((error) => showError(errorTargetForCurrentForm(), error));
}
