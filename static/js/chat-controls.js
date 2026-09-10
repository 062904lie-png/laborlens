/* Presentation only: source metadata and the existing answer handlers remain authoritative. */
(function () {
  const paths = {
    copy:'<rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V3H3v13h5"/>',
    up:'<path d="M7 10h-4v11h4M7 20h11a2 2 0 0 0 2-2l1-7a2 2 0 0 0-2-2h-6l1-5c0-3-3-3-4-1l-3 7Z"/>',
    down:'<path d="M7 14H3V3h4M7 4h11a2 2 0 0 1 2 2l1 7a2 2 0 0 1-2 2h-6l1 5c0 3-3 3-4 1l-3-7Z"/>',
    flag:'<path d="M5 22V3m0 1c5-5 9 5 15 0v11c-6 5-10-5-15 0"/>',
    external:'<path d="M14 3h7v7M21 3 10 14M10 3H3v18h18v-7"/>'
  };
  function icon(name){return `<svg class="ll-action-icon" viewBox="0 0 24 24" aria-hidden="true">${paths[name]||paths.external}</svg>`;}
  // Normalize authority names without changing their numbers (297 must not match 2970).
  function keys(text){return (String(text||'').toLowerCase().replace(/republic\s+act|r\.?\s*a\.?/g,'ra').replace(/department\s+order|d\.?\s*o\.?/g,'do').replace(/article|art\./g,'article').match(/\b(?:ra|do|article|section)\s*(?:no\.?\s*)?\d+(?:[-–]\d+)?/g)||[]).map(s=>s.replace(/no\.?|\s/g,'').replace('–','-'));}
  function matchSource(citation,sources){
    const wanted=keys(citation);if(!wanted.length)return -1;
    const matches=sources.map((source,index)=>{const text=[source.citation,source.law_number,source.article_section,source.document_title,source.source,source.article_title].filter(Boolean).join(' ');return {index,text,keys:keys(text)};}).filter(item=>wanted.every(key=>item.keys.includes(key)) && (!/labor\s+code/i.test(citation)||/labor\s+code/i.test(item.text)));
    if(matches.length===1)return matches[0].index;
    // Several excerpts of the same authority are safe only when they share the same saved URL.
    if(matches.length>1){const urls=new Set(matches.map(m=>sources[m.index].source_url||''));if(urls.size===1 && !urls.has(''))return matches[0].index;}
    return -1;
  }
  function linkLegalBasis(row,sources){
    const section=row.querySelector('.answer-section-legal');if(!section)return;
    const walker=document.createTreeWalker(section,NodeFilter.SHOW_TEXT);
    const nodes=[];while(walker.nextNode())if(!walker.currentNode.parentElement.closest('a,button,.answer-label'))nodes.push(walker.currentNode);
    const pattern=/\b(?:(?:Labor Code,?\s*)?(?:Article|Art\.)\s*\d+|(?:RA|R\.A\.|Republic Act)\s*(?:No\.?\s*)?\d+(?:,?\s*Section\s*\d+)?|(?:DO|D\.O\.|Department Order)\s*(?:No\.?\s*)?\d+(?:[-–]\d+)?)/gi;
    nodes.forEach(node=>{
      const text=node.textContent;const fragment=document.createDocumentFragment();let offset=0,changed=false;
      for(const match of text.matchAll(pattern)){
        const index=matchSource(match[0],sources);if(index<0)continue;
        fragment.append(document.createTextNode(text.slice(offset,match.index+match[0].length)));
        const url=window.safeSourceUrl(sources[index].source_url||'');
        const link=document.createElement(url?'a':'button');link.className='legal-source-link';link.title='View source';link.dataset.tooltip='View source';link.setAttribute('aria-label','View source for '+match[0]);link.innerHTML=icon('external');
        if(url){link.href=url;link.target='_blank';link.rel='noopener noreferrer';}
        else{link.type='button';link.addEventListener('click',()=>{
          const list=row.querySelector('.source-hidden-passages');const item=list?.children[index];if(!item)return;
          list.classList.add('show');const toggle=row.querySelector('[data-view-label]');if(toggle){toggle.setAttribute('aria-expanded','true');toggle.classList.add('saved');}
          const passage=item.querySelector('details');if(passage)passage.open=true;
          item.tabIndex=-1;item.focus({preventScroll:true});item.scrollIntoView({block:'nearest',behavior:'smooth'});
        });}
        fragment.append(link);offset=match.index+match[0].length;changed=true;
      }
      if(changed){fragment.append(document.createTextNode(text.slice(offset)));node.replaceWith(fragment);}
    });
  }
  window.LaborLensChatControls={icon,keys,matchSource,linkLegalBasis};
})();
