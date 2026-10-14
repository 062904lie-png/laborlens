/* Invalidate the admin review view without sharing chat contents or account IDs. */
(() => {
  const eventName='laborlens-chat-deleted',storageKey='ll_chat_change';
  let channel=null,pending=false,timer=null;
  try{if(typeof BroadcastChannel==='function')channel=new BroadcastChannel('laborlens-chat-changes');}catch(_){}
  async function refresh(){
    if(pending||document.visibilityState!=='visible'||!window.AdminReviews?.refresh)return;
    pending=true;try{await window.AdminReviews.refresh();}finally{pending=false;}
  }
  function schedule(){clearTimeout(timer);timer=setTimeout(()=>{void refresh();},250);}
  function notifyDeleted(){
    window.dispatchEvent(new Event(eventName));
    channel?.postMessage({type:'chat_deleted'});
    try{localStorage.setItem(storageKey,Date.now()+':'+Math.random());}catch(_){}
  }
  if(channel)channel.onmessage=event=>{if(event.data?.type==='chat_deleted')schedule();};
  window.addEventListener(eventName,schedule);
  window.addEventListener('storage',event=>{if(event.key===storageKey)schedule();});
  window.addEventListener('focus',schedule);
  document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible')schedule();});
  // Covers another browser/device/origin. Only the visible authenticated review
  // queue (or its open detail view) performs a refresh; no model calls are made.
  setInterval(()=>{void refresh();},15000);
  window.LaborLensChatChanges={notifyDeleted};
})();
