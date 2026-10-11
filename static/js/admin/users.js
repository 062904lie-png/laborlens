/* Ordinary-user administration only. Administrator records remain untouched. */
let userOffset=0,userVersion=0;
const ordinaryUser=user=>!['admin','administrator','super admin','super_admin'].includes(String(user?.role||'user').trim().toLowerCase());

function setupUsers(){
 const page=document.getElementById('tab-users');
 page.innerHTML=`<div class="page-hdr"><div class="page-hdr-left"><h1>Users</h1><p>Manage user accounts and access to LaborLens. Personal details are hidden. Inactive accounts cannot use the chatbot.</p></div></div><div id="user-stats" class="users-summary"></div><form class="workbench-filters users-filter-panel" id="users-filter"><label>Search<input id="user-search" type="search" placeholder="Search user ID..."/></label><label>Role<select id="users-role"><option value="">All roles</option></select></label><label>Status<select id="users-status"><option value="">All statuses</option><option value="active">Active</option><option value="inactive">Inactive</option></select></label><button class="btn btn-primary" type="submit">Filter</button><button class="btn" type="reset">Reset</button></form><section class="users-table-card"><div id="user-table"></div><div class="workbench-pager" id="user-pager"></div></section>`;
 const form=page.querySelector('form');
 form.onsubmit=event=>{event.preventDefault();userOffset=0;loadUsers();};
 form.onreset=()=>{userOffset=0;setTimeout(loadUsers,0);};
}

async function loadUsers(){
 if(!document.getElementById('user-table'))setupUsers();
 const version=++userVersion,table=document.getElementById('user-table');
 table.innerHTML='<div class="loading">Loading users...</div>';
 try{
  const params=new URLSearchParams({q:document.getElementById('user-search')?.value.trim()||'',role:document.getElementById('users-role')?.value||'',status:document.getElementById('users-status')?.value||'',offset:String(userOffset),limit:String(adminPageSize())});
  const result=await AdminPages.api('/admin/users?'+params);
  if(version!==userVersion)return;
  const records=(Array.isArray(result?.items)?result.items:[]).filter(ordinaryUser);
  allUsers=records;
  const roles=Array.isArray(result?.roles)?result.roles:[];
  document.getElementById('users-role').innerHTML=`<option value="">All roles</option>${roles.map(role=>`<option value="${esc(role)}">${esc(role==='user'?'User':role)}</option>`).join('')}`;
  document.getElementById('users-role').value=params.get('role');
  const stats=result?.stats||{};
  document.getElementById('user-stats').innerHTML=AdminUI.cards([
   ['Total Users',Number(stats.total)||0,'blue','users'],
   ['Active Users',Number(stats.active)||0,'green','users'],
   ['Inactive Users',Number(stats.inactive)||0,'red','users']
  ]);
  if(userOffset>=Number(result?.total||0)&&userOffset){userOffset=0;return loadUsers();}
  renderUsers(records,Number(result?.total||0),Boolean(params.get('q')||params.get('role')||params.get('status')));
 }catch(error){
  if(version===userVersion)table.innerHTML='<div class="err-msg">Users could not be loaded. <button class="btn" onclick="loadUsers()">Retry</button></div>';
 }
}

function filterUsers(){
 userOffset=0;
 return loadUsers();
}

function renderUsers(users,total,filtered=false){
 const table=document.getElementById('user-table');
 if(!users.length){
  table.innerHTML=`<div class="users-empty"><p>${filtered?'No users match your current filters.':'No user accounts found.'}</p>${filtered?'<button class="btn btn-sm" id="users-reset-empty">Reset Filters</button>':''}</div>`;
  document.getElementById('users-reset-empty')?.addEventListener('click',()=>document.getElementById('users-filter').reset());
  document.getElementById('user-pager').innerHTML='';
  return;
 }
 const pageSize=adminPageSize(),visible=users;
 table.innerHTML=AdminUI.table(['#','User','Role','Status','Preferred Language','Date Registered','Actions'],visible.map((user,index)=>[
  userOffset+index+1,esc(user.anonymous_user||'Anonymous User'),AdminUI.badge(user.role==='user'?'User':user.role,'blue'),AdminUI.badge(user.is_active?'Active':'Inactive',user.is_active?'green':'red'),esc(user.language_preference?languageLabel(user.language_preference):'Not set'),esc(dateLabel(user.created_at)),`<button class="btn btn-sm" data-user-index="${index}">View</button>`
 ]),'users-records-table');
 table.querySelectorAll('[data-user-index]').forEach(button=>button.onclick=()=>showUser(visible[Number(button.dataset.userIndex)]));
 AdminUI.pager(document.getElementById('user-pager'),userOffset,total,next=>{userOffset=next;loadUsers();},pageSize);
}

function showUser(user){
 if(!user)return;
 const U=AdminUI;
 U.drawer('User Details',`<section class="user-detail-summary"><h3>${esc(user.anonymous_user||'Anonymous User')}</h3>${U.badge(user.is_active?'Active':'Inactive',user.is_active?'green':'red')}${U.badge(user.role==='user'?'User':user.role,'blue')}<div id="user-detail-tabs"></div></section>`, `<button class="btn btn-danger" id="user-access-button">${user.is_active?'Disable Account':'Reactivate Account'}</button>`);
 U.tabs(document.getElementById('user-detail-tabs'),[['Overview',U.table(['Field','Value'],[['Identity',esc(user.anonymous_user||'Anonymous User')],['Role',esc(user.role==='user'?'User':user.role)],['Account status',esc(user.is_active?'Active':'Inactive')],['Preferred language',esc(user.language_preference?languageLabel(user.language_preference):'Not set')],['Date registered',esc(dateLabel(user.created_at))]])]]);
  document.getElementById('user-access-button').addEventListener('click',async()=>{if(await setUserAccess(Number(user.id),!user.is_active))closeDocModal();});
}

async function toggleRole(id, cur) {
  const next = cur === 'admin' ? 'user' : 'admin';
  if (!confirm(`Update user privileges to "${next.toUpperCase()}"?`)) return;
  try {
    const r = await apiFetch(`/admin/users/${id}/role?role=${next}`, {method:'PUT'});
    if (!r.ok) throw new Error('Update failed.');
    toast(`Privileges updated to ${next.toUpperCase()}`);
    loadUsers(); loadOverview();
  } catch(e) { toast(e.message, true); }
}
