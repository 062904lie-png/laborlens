/* Small registry shared by page-specific controllers. Classic scripts support file:// previews. */
window.AdminPages=(()=>{
  const pages={};
  const table=(headers,rows)=>`<div class="tbl-wrap"><table><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.length?rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join('')}</tr>`).join(''):`<tr><td colspan="${headers.length}" class="empty">No matching records.</td></tr>`}</tbody></table></div>`;
  async function api(path,opts={}){const r=await apiFetch(path,opts);if(!r.ok)throw Error('Unable to load or update this page. Check your session and backend.');return r.json();}
  function register(id,title,description,loader){
    const page=document.createElement('section');page.id=`tab-${id}`;page.className='workbench-tab';
    page.innerHTML=`<div class="page-hdr"><div class="page-hdr-left"><h1>${esc(title)}</h1><p>${esc(description)}</p></div></div><form class="workbench-filters"><label>Search<input type="search" maxlength="200"/></label><button class="btn btn-primary">Search / Refresh</button></form><div class="page-results"></div>`;
    document.querySelector('.content-inner').append(page);
    let generation=0;
    const load=async()=>{const v=++generation;const result=page.querySelector('.page-results');result.innerHTML='<div class="loading">Loading...</div>';try{const output=await loader(page.querySelector('input[type="search"]')?.value||'',page);if(v===generation&&output!==undefined){result.innerHTML=typeof output==='string'?output:output.html;output.mount?.(result);}}catch(e){if(v===generation)result.innerHTML=`<div class="err-msg">${esc(e.message)}</div>`;}};
    const form=page.querySelector('form');const reset=document.createElement('button');reset.type='reset';reset.className='btn';reset.textContent='Reset';form.append(reset);form.onreset=()=>setTimeout(()=>{page.dataset.offset='0';load();},0);
    form.onsubmit=e=>{e.preventDefault();page.dataset.offset='0';load();};pages[id]=load;
    return page;
  }
  function localPage(rows,page,headers,row){const pageSize=adminPageSize(),offset=Math.min(Number(page.dataset.offset||0),Math.max(0,Math.floor((rows.length-1)/pageSize)*pageSize)),visible=rows.slice(offset,offset+pageSize);return {html:table(headers,visible.map(row)),mount(result){const controls=document.createElement('div');controls.className='workbench-pager';controls.innerHTML=`<span>${rows.length?offset+1:0}–${Math.min(offset+pageSize,rows.length)} of ${rows.length}</span><button class="btn btn-sm" ${offset?'':'disabled'}>Previous</button><button class="btn btn-sm" ${offset+pageSize<rows.length?'':'disabled'}>Next</button>`;controls.querySelectorAll('button').forEach((button,index)=>button.onclick=()=>{page.dataset.offset=String(offset+(index?pageSize:-pageSize));pages[page.id.replace('tab-','')]();});result.append(controls);}};}
  return {register,table,api,localPage,open:id=>pages[id]?.()};
})();
