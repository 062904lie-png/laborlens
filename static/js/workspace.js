/* Line icons for the consultation shell; all buttons retain their original handlers. */
(function () {
  const paths = {
    chat:'<path d="M21 11.5a9 9 0 0 1-9 9 10 10 0 0 1-4-.9L3 21l1.4-4.8A9 9 0 1 1 21 11.5Z"/>',
    calculator:'<rect x="5" y="2" width="14" height="20" rx="1.5"/><path d="M8 5h8v4H8zM8 12h1m3 0h1m3 0h.1M8 15h1m3 0h1m3 0h.1M8 18h1m3 0h1m3 0h.1"/>',
    gift:'<rect x="3" y="7" width="18" height="5" rx="1"/><path d="M5 12v10h14V12M12 7v15"/><path d="M12 7H8a3 3 0 1 1 3-3zm0 0h4a3 3 0 1 0-3-3z"/>',
    clock:'<circle cx="12" cy="12" r="10"/><path d="M12 5v7l5 3"/>',
    calendar:'<rect x="3" y="5" width="18" height="17" rx="2"/><path d="M7 2v6m10-6v6M3 11h18M7 15h1m4 0h1m4 0h1M7 18h1m4 0h1"/>',
    users:'<circle cx="8" cy="6" r="3"/><circle cx="17" cy="6" r="3"/><path d="M2 22v-5a6 6 0 0 1 12 0v5m3-11a5 5 0 0 1 5 5v6"/>',
    file:'<path d="M14 2H5v20h14V7zM14 2v6h5M8 12h8m-8 4h8m-8 3h5"/>',
    arrow:'<path d="m9 5 7 7-7 7"/>',
    chevron:'<path d="m5 9 7 6 7-6"/>',
    plus:'<path d="M12 3v18M3 12h18"/>',
    search:'<circle cx="10" cy="10" r="7"/><path d="m15 15 6 6"/>',
    filter:'<path d="M2 5h12m4 0h4M2 12h4m4 0h12M2 19h12m4 0h4"/><circle cx="16" cy="5" r="2"/><circle cx="8" cy="12" r="2"/><circle cx="16" cy="19" r="2"/>',
    sparkle:'<path d="M10 3c0 6-2 8-7 9 5 1 7 3 7 9 1-6 3-8 8-9-5-1-7-3-8-9Zm9-2v6m-3-3h6"/>',
    attach:'<path d="m8 13 7-7a3 3 0 0 1 4 4l-9 9a5 5 0 0 1-7-7L13 2a4 4 0 0 1 6 6l-9 9a2 2 0 0 1-3-3l8-8"/>',
    send:'<path d="m22 2-7 20-4-9-9-4zM11 13 22 2"/>',
    shield:'<path d="m12 2 8 3v7c0 5-4 8-8 10-4-2-8-5-8-10V5zM8 12l3 3 5-6"/>',
    settings:'<path d="m9 3 1-2h4l1 2 3 2 2 .1 2 3-1 2v4l1 2-2 3-2 .1-3 2-1 2h-4l-1-2-3-2-2-.1-2-3 1-2v-4l-1-2 2-3L6 5Z"/><circle cx="12" cy="12" r="4"/>',
    logout:'<path d="M10 3H3v18h7m4-15 6 6-6 6M7 12h13"/>'
  };
  function icon(name) { return '<svg class="ll-icon" viewBox="0 0 24 24" aria-hidden="true">'+paths[name]+'</svg>'; }
  function fill(root, selector, name) {
    root.querySelectorAll(selector).forEach(el => { if (!el.querySelector('svg')) el.innerHTML=icon(name); });
  }
  function render() {
    const app = document.getElementById('app');
    if (!app) return;
    fill(app, '.mode-tab-icon', 'chat');
    app.querySelectorAll('.consultation-actions .mode-tab:nth-child(2) .mode-tab-icon, [data-benefits-icon]').forEach(el => {
      if (!el.dataset.calculatorIcon) { el.innerHTML=icon('calculator'); el.dataset.calculatorIcon='1'; }
    });
    ['gift','clock','calendar','users','file'].forEach((name,i)=>fill(app,
      '.topic-card:nth-child('+(i+1)+') :is(.topic-icon,.tc-icon)', name));
    fill(app,'.mode-tab-arrow,.topic-arrow,.tc-arrow','arrow');
    fill(app,'.common-topics-title > span,.topics-header > span,.input-sparkle','sparkle');
    fill(app,'.tb-user-chevron,.chevron','chevron');
    fill(app,'.sess-icon,.session-icon','chat');
    fill(app,'#send-btn','send');
    fill(app,'#attach-btn,.attach-btn','attach');
    fill(app,'.disclaimer-shield,.shield','shield');
    fill(app,'.sidebar-nav-btn:not(.logout) > span,.sidebar-btn:first-child > span','settings');
    fill(app,'.sidebar-nav-btn.logout > span,.sidebar-btn:last-child > span','logout');
    app.querySelectorAll('.sidebar-brand-mark').forEach(el=>{
      if (el.querySelector('svg')) return;
      el.innerHTML='<svg viewBox="0 0 68 84" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2"><path d="M34 2C24 9 13 12 5 15v38c0 14 19 24 29 29 10-5 29-15 29-29V15C55 12 44 9 34 2Z"/><path d="M34 18v45m-13 4h26M16 27h36M34 22l-9 5m9-5 9 5M19 29l-9 22h18zm30 0-9 22h18z"/><path d="M10 51q9 10 18 0m12 0q9 10 18 0M29 63h10" fill="currentColor"/></svg>';
    });
    app.querySelectorAll('.sb-new-btn').forEach(el=>{
      if (el.dataset.referenceIcon) return;
      el.innerHTML='<span class="ll-plus">'+icon('plus')+'</span>New Consultation'+icon('arrow');
      el.dataset.referenceIcon='1';
    });
    app.querySelectorAll('.session-search,.session-search-wrapper').forEach(el=>{
      if (el.querySelector('.ll-search-icon')) return;
      el.querySelectorAll('.search-icon,.filter-icon').forEach(old=>old.remove());
      el.insertAdjacentHTML('beforeend','<span class="ll-search-icon">'+icon('search')+'</span><span class="ll-filter-icon">'+icon('filter')+'</span>');
    });
    const composer = app.querySelector('#input-box');
    if (composer && !composer.querySelector('.input-sparkle')) composer.insertAdjacentHTML('afterbegin','<span class="input-sparkle" aria-hidden="true">'+icon('sparkle')+'</span>');
  }
  document.addEventListener('DOMContentLoaded',()=>{
    render();
    const observer = new MutationObserver(render);
    const app = document.getElementById('app');
    if (app) observer.observe(app,{childList:true,subtree:true});
  });
})();
