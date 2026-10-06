import { initializeApp } from 'https://www.gstatic.com/firebasejs/12.0.0/firebase-app.js';
import {
  getAuth,
  GoogleAuthProvider,
  signInWithPopup,
} from 'https://www.gstatic.com/firebasejs/12.0.0/firebase-auth.js';

const firebaseConfig = {
  apiKey: 'AIzaSyACbScu2F1HdC_09TUUvEzE2slpEvOKJLA',
  authDomain: 'laborlens-31a9c.firebaseapp.com',
  projectId: 'laborlens-31a9c',
  storageBucket: 'laborlens-31a9c.firebasestorage.app',
  messagingSenderId: '373394325577',
  appId: '1:373394325577:web:5aafdc3b9d114e6c8bf98',
  measurementId: 'G-FEGGZ5PX88',
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();
provider.setCustomParameters({ prompt: 'select_account' });

window.LaborLensFirebase = {
  async signIn(apiBase) {
    const { user } = await signInWithPopup(auth, provider);
    const idToken = await user.getIdToken();
    const response = await fetch(`${apiBase.replace(/\/$/, '')}/auth/firebase`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id_token: idToken }),
    });
    let result;
    try {
      result = await response.json();
    } catch {
      result = {};
    }
    if (!response.ok) {
      throw new Error(result.detail || `LaborLens sign-in failed (${response.status}).`);
    }
    if (typeof window.saveAuth !== 'function') {
      throw new Error('LaborLens could not finish sign-in. Refresh and try again.');
    }
    window.saveAuth(result);
  },
};
