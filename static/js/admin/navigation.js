/* Consolidated admin page. */
window.AdminShell=(()=>{
 const pages={overview:['Dashboard','home'],knowledge:['Knowledge Base','folder'],reviews:['Answer Reviews','chat'],reports:['Reports','chart'],users:['Users','users'],activity:['Admin Activity','file'],settings:['Settings','settings']};
 const tabs={knowledge:[['knowledge','Documents'],['categories','Categories'],['coverage','Coverage'],['faqs','FAQs']],reviews:[['reviews','Review Queue'],['corrections','Corrections'],['playground','Test Chatbot']],reports:[['analytics','Overview'],['feedback','Feedback'],['performance','Performance']]};
 const owners=Object.fromEntries(Object.entries(tabs).flatMap(([owner,items])=>items.map(([id])=>[id,owner])));
 const aliases={dashboard:'overview','knowledge-base':'knowledge','answer-reviews':'reviews','chat-reviews':'reviews',conversations:'reviews','reported-answers':'reviews','source-review':'knowledge','legal-source-review':'knowledge','system-health':'performance',queries:'reviews','user-feedback':'feedback','activity-logs':'activity','admin-activity':'activity'};
 const nav=document.getElementById('left-nav');nav.innerHTML=`<div class="admin-sidebar-brand"><span aria-hidden="true">⚖</span><div><strong>Labor<b>Lens</b></strong><small>ADMIN</small></div></div><div class="admin-nav-scroll"></div><div class="admin-nav-footer"><a class="end-btn" href="index.html" target="_blank" rel="noopener">View Website</a><button type="button" class="end-btn" onclick="doLogout()">Sign Out</button></div>`;
 const groups=[['',Object.keys(pages)]];
 const shell=document.createElement('header');shell.id='admin-section-header';shell.innerHTML='<h1 id="admin-section-title"></h1><div class="detail-tabs" id="admin-section-tabs" role="tablist" aria-label="Section tabs"></div>';document.querySelector('.content-inner').prepend(shell);
 nav.querySelector('.admin-nav-scroll').innerHTML=groups.map(([title,ids])=>(title?`<div class="nav-section-lbl">${title}</div>`:'')+ids.map(id=>`<button type="button" class="nav-item ${id==='overview'?'active':''}" data-tab="${id}"><span class="nav-icon">${AdminUI.icon(pages[id][1])}</span>${pages[id][0]}</button>`).join('')).join('');nav.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>go(b.dataset.tab));
 for(const id of ['source-review','reports','conversations','queries'])document.getElementById('tab-'+id)?.remove();
 const top=document.getElementById('topbar'),search=document.createElement('form');search.className='admin-global-search';search.innerHTML=`${AdminUI.icon('search')}<input aria-label="Global admin search" placeholder="Search questions, users, or documents..." maxlength="200"/><button type="submit" aria-label="Search">↵</button>`;top.querySelector('.tb-spacer').before(search);search.onsubmit=e=>{e.preventDefault();const query=search.querySelector('input').value;AdminReviews.setSearch(query);go('reviews');};
 search.onsubmit=async e=>{e.preventDefault();const q=search.querySelector('input').value.trim();if(!q)return;AdminUI.drawer('Search Results','<div class="loading">Searching...</div>');try{const [reviews,docs,users]=await Promise.all([AdminPages.api('/admin/answer-reviews?q='+encodeURIComponent(q)+'&limit=10'),AdminPages.api('/kb/documents'),AdminPages.api('/admin/users')]);const term=q.toLowerCase();const ds=docs.filter(d=>[d.document_title,d.original_filename,d.issuing_agency].some(v=>String(v||'').toLowerCase().includes(term))).slice(0,10);const us=users.filter(u=>String(u.anonymous_user||'').toLowerCase().includes(term)).slice(0,10);AdminUI.drawer('Search Results',AdminUI.panel('Questions',AdminUI.table(['Question','Action'],reviews.items.map(r=>[esc(r.question),`<button class="btn btn-sm" data-search-review="${Number(r.id)}">Review</button>`])))+AdminUI.panel('Documents',AdminUI.table(['Document','Action'],ds.map(d=>[documentTitleHtml(documentDisplayTitle(d)),`<button class="btn btn-sm" data-search-doc="${Number(d.id)}">View</button>`])))+AdminUI.panel('Users (masked)',AdminUI.table(['User','Action'],us.map(u=>[esc(u.anonymous_user),`<button class="btn btn-sm" data-search-user="${Number(u.id)}">View</button>`]))));document.querySelectorAll('[data-search-review]').forEach(b=>b.onclick=()=>{go('reviews');AdminReviews.open(Number(b.dataset.searchReview));});document.querySelectorAll('[data-search-doc]').forEach(b=>b.onclick=()=>{closeDocModal();go('knowledge');viewDoc(Number(b.dataset.searchDoc));});document.querySelectorAll('[data-search-user]').forEach(b=>b.onclick=async()=>{closeDocModal();go('users');await loadUsers();showUser(Number(b.dataset.searchUser));});}catch(e){AdminUI.drawer('Search Results','<div class="err-msg">Search could not be completed. Please try again.</div>');}};
 try{document.getElementById('dashboard').dataset.density=localStorage.getItem('laborlens_admin_density')||'comfortable';}catch(e){}
 const bell=document.createElement('button');bell.type='button';bell.className='admin-notifications';bell.setAttribute('aria-label','Open reported answers');bell.innerHTML=AdminUI.icon('bell');bell.onclick=()=>go('reviews','reported');top.querySelector('.tb-user-chip').before(bell);
 async function go(id,filter='',updateHash=true,tab=''){
  const requested=id;id=aliases[id]||id;let leaf=id;
  if(id==='reports')leaf=tab&&tabs.reports.some(([key])=>key===tab)?tab:'analytics';
  else if(tabs[id]&&tab&&tabs[id].some(([key])=>key===tab))leaf=tab;
  const owner=owners[leaf]||leaf;if(!pages[owner])return go('overview');
  if(['reported-answers'].includes(requested))filter='reported';
  if(['source-review','legal-source-review'].includes(requested))filter='needs_attention';
  if(requested==='queries')filter='unresolved';
  document.querySelectorAll('.content-inner > [id^="tab-"]').forEach(p=>p.classList.toggle('tab-active',p.id==='tab-'+leaf));
  nav.querySelectorAll('[data-tab]').forEach(b=>{b.classList.toggle('active',b.dataset.tab===owner);b.setAttribute('aria-current',b.dataset.tab===owner?'page':'false');});
  document.getElementById('admin-section-title').textContent=pages[owner][0];
  const bar=document.getElementById('admin-section-tabs');bar.innerHTML=(tabs[owner]||[]).map(([key,title])=>`<button type="button" role="tab" aria-selected="${key===leaf}" class="${key===leaf?'active':''}" data-leaf="${key}">${title}</button>`).join('');
  bar.querySelectorAll('button').forEach(b=>b.onclick=()=>go(b.dataset.leaf));
  bar.hidden=!tabs[owner];shell.hidden=owner==='overview';
  closeAdminNav();document.getElementById('main').scrollTop=0;
  if(updateHash){const params=new URLSearchParams();if(tabs[owner])params.set('tab',leaf);if(filter)params.set('filter',filter);history.replaceState(null,'','#'+owner+(params.size?'?'+params:''));}
  if(!token)return;
  if(leaf==='reviews')return filter?AdminReviews.setFilter(filter):AdminReviews.load();
  if(leaf==='knowledge'){AdminKnowledge.setReviewFilter(filter);await loadDocuments();if(filter==='add')AdminUpload.open();return;}
  if(leaf==='overview')return loadOverview();
  if(leaf==='analytics')return loadAnalytics();
  if(leaf==='users')return loadUsers();
  return AdminPages.open(leaf);
 }
 function route(){const [raw,query='']=location.hash.slice(1).split('?');go(raw||'overview',new URLSearchParams(query).get('filter')||new URLSearchParams(query).get('review')||'',false,new URLSearchParams(query).get('tab')||'');}
 window.addEventListener('hashchange',route);
 const originalInit=initDashboard;initDashboard=function(){originalInit();route();};
 return {go,route};
})();
