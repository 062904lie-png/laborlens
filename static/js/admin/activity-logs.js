/* Read-only, paginated audit trail. Details are already redacted by the API. */
function activityLabel(value){return String(value||'Not recorded').replaceAll('_',' ').replace(/\b\w/g,letter=>letter.toUpperCase());}
function activityDate(value){const date=new Date(value);return Number.isNaN(date.getTime())?String(value||'Not recorded'):date.toLocaleString(undefined,{month:'short',day:'numeric',year:'numeric',hour:'numeric',minute:'2-digit'});}
function activityDetails(value){if(value&&typeof value==='object')return value;try{return JSON.parse(String(value||'{}'));}catch(_){return {detail:String(value||'Not recorded')};}}
function activityDetailRows(value){return Object.entries(activityDetails(value)).map(([key,item])=>[activityLabel(key),esc(typeof item==='string'?item:JSON.stringify(item))]);}
function activitySummary(value){const rows=activityDetailRows(value);return rows.length?rows.slice(0,2).map(([key,item])=>`<span><b>${key}:</b> ${item}</span>`).join(' '):'No extra details';}
function activityOptions(rows,selected,empty){return `<option value="">${esc(empty)}</option>${rows.map(row=>`<option value="${esc(row.label)}" ${String(row.label)===String(selected)?'selected':''}>${esc(activityLabel(row.label))} (${Number(row.count||0).toLocaleString()})</option>`).join('')}`;}

const activityPage=AdminPages.register('activity','Admin Activity','View system activities, admin actions, and important events in LaborLens.',async(search,page)=>{
 const U=AdminUI,form=page.querySelector('form'),params=new URLSearchParams(new FormData(form));params.set('q',search);params.set('limit',adminPageSize());const offset=Number(page.dataset.offset||0);params.set('offset',offset);
 const data=await AdminPages.api('/admin/audit-page?'+params),groups=Array.isArray(data.groups)?data.groups:[],actions=Array.isArray(data.actions)?data.actions:[];
 const action=String(form.elements.action?.value||''),module=String(form.elements.module?.value||'');
 if(form.elements.action)form.elements.action.innerHTML=activityOptions(actions,action,'All actions');
 if(form.elements.module)form.elements.module.innerHTML=activityOptions(groups,module,'All sections');
 const cards=[['Total Activities',data.total,'blue','file'],...groups.slice(0,4).map((row,index)=>[activityLabel(row.label),row.count,['green','purple','gold','red'][index],'file'])];
 const rows=(data.items||[]).map((row,index)=>[offset+index+1,esc(activityDate(row.created_at)),esc(row.admin_username||'System'),U.badge(activityLabel(row.action)),esc(activityLabel(row.target_type)),`<span class="activity-detail-summary">${activitySummary(row.details)}</span>`,`<button class="btn btn-sm" data-activity="${Number(row.id)}">View</button>`]);
 return {html:U.cards(cards)+U.table(['#','Date & Time','User','Action','Section','Details','Action'],rows,'activity-records-table')+'<div class="workbench-pager" id="activity-pager"></div>',mount(host){
  U.pager(host.querySelector('#activity-pager'),offset,Number(data.total||0),next=>{page.dataset.offset=String(next);AdminPages.open('activity');},adminPageSize());
  host.querySelectorAll('[data-activity]').forEach(button=>button.onclick=()=>{const row=(data.items||[]).find(item=>Number(item.id)===Number(button.dataset.activity));if(!row)return;U.drawer('Activity Details',`<h3>${esc(activityLabel(row.action))}</h3>`+U.table(['Field','Value'],[['Date & Time',esc(activityDate(row.created_at))],['User',esc(row.admin_username||'System')],['Section',esc(activityLabel(row.target_type))],['Target',esc(row.target_id||'Not recorded')],...activityDetailRows(row.details)]));});
 }};
});

const activityFilters=activityPage.querySelector('form');
activityFilters.querySelector('input[type=search]').placeholder='Search activity details...';
activityFilters.querySelector('button[type=submit]').textContent='Filter';
activityFilters.querySelector('button[type=submit]').insertAdjacentHTML('beforebegin',`<label>Action<select name="action"><option value="">All actions</option></select></label><label>Admin username<input name="user" placeholder="Search username"/></label><label>Section<select name="module"><option value="">All sections</option></select></label><label>Date range<select name="days"><option value="30" ${adminReportPeriod()==='30'?'selected':''}>Last 30 days</option><option value="7" ${adminReportPeriod()==='7'?'selected':''}>Last 7 days</option></select></label>`);
