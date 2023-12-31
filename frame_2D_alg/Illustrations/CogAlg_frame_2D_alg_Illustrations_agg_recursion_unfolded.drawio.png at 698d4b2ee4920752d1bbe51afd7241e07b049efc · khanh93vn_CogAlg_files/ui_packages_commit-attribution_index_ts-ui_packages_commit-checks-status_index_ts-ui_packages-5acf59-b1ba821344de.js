"use strict";(globalThis.webpackChunk=globalThis.webpackChunk||[]).push([["ui_packages_commit-attribution_index_ts-ui_packages_commit-checks-status_index_ts-ui_packages-5acf59"],{15706:(e,t,o)=>{o.d(t,{D:()=>b,C:()=>z});var r=o(85893),a=o(57294),n=o(78212),i=o(42483),l=o(73290),s=o(97011);function c(e){return e.path?.startsWith("/apps/")??!1}var d=o(65379);function p({renderTooltip:e,author:t,children:o}){return!1===e?(0,r.jsx)(r.Fragment,{children:o}):(0,r.jsx)(d.Z,{"aria-label":`commits by ${t.login}`,direction:"se",children:o})}try{p.displayName||(p.displayName="AuthorTooltip")}catch{}var u=o(9996),h=o.n(u),m=o(67294);let f={fontWeight:"bold",fontColor:"fg.default",includeTooltip:!0,avatarSize:void 0},x=(0,m.createContext)(f);function g({authorSettings:e,children:t}){let o=h()(f,e??{});return(0,r.jsx)(x.Provider,{value:o,children:t})}function y(){return(0,m.useContext)(x)||f}try{x.displayName||(x.displayName="AuthorSettingsContext")}catch{}try{g.displayName||(g.displayName="AuthorSettingsProvider")}catch{}function b({author:e,repo:t,sx:o={}}){let d=y();if(!e)return null;let u=(0,r.jsx)(a.O,{"aria-label":`${e.login||"author"}`,src:e.avatarUrl,alt:`${e.login||"author"}`,sx:{mr:2,mt:"-1px",ml:"1px"},size:d.avatarSize,square:c(e)});return(0,r.jsxs)(i.Z,{sx:{display:"flex",flexDirection:"row",alignItems:"center",...o},"data-testid":"author-avatar",children:[e.path?(0,r.jsx)(l.Z,{href:e.path,"data-testid":"avatar-icon-link","data-hovercard-url":e.login?(0,n.zP)({owner:e.login}):void 0,children:u}):u,e.login?(0,r.jsx)(p,{author:e,renderTooltip:d.includeTooltip,children:(0,r.jsx)(l.Z,{muted:!0,href:(0,n.OI)({repo:t,author:e.login}),"aria-label":`commits by ${e.login}`,sx:{fontWeight:d.fontWeight,whiteSpace:"nowrap",color:d.fontColor,"&:hover":{color:d.fontColor,textDecoration:"underline"}},children:e.login})}):(0,r.jsx)(s.Z,{sx:{fontWeight:d.fontWeight,whiteSpace:"nowrap",color:d.fontColor},children:e.displayName})]})}try{b.displayName||(b.displayName="AuthorAvatar")}catch{}var j=o(52516),v=o(79902),C=o(43535);function k({authors:e,repo:t}){let o=e.length,[a,n]=(0,m.useState)(!1),i=(0,m.useRef)(null);return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsxs)(l.Z,{as:"button","aria-label":`Show ${o} authors`,"data-testid":"authors-dialog-anchor",onClick:()=>{n(!0)},sx:{mx:1},ref:i,muted:!0,children:[o," ","people"]}),a&&(0,r.jsx)(C.V,{title:`${o} authors`,onClose:()=>{n(!1),setTimeout(()=>i.current?.focus(),25)},width:"medium",height:o>=12?"small":"auto",renderBody:()=>(0,r.jsx)(j.S,{sx:{overflowY:"auto",py:2},"data-testid":"contributor-dialog-list",children:e.map((e,o)=>(0,r.jsx)(w,{author:e,repo:t},`${e.login}_${o}`))})})]})}function w({author:e,repo:t}){return(0,r.jsxs)(j.S.LinkItem,{sx:{display:"flex",flexDirection:"row",fontSize:1,py:2,color:"fg.default","&:hover":{backgroundColor:"canvas.subtle"}},"data-testid":"contributor-dialog-row",href:(0,n.OI)({repo:t,author:e.login??""}),children:[(0,r.jsx)(a.O,{src:e.avatarUrl,alt:e.login??e.displayName,sx:{mr:2},"aria-hidden":"true",square:c(e)}),(0,r.jsx)(v.Z,{inline:!0,title:e.login??e.displayName??"",children:e.login??e.displayName})]})}try{k.displayName||(k.displayName="AuthorsDialog")}catch{}try{w.displayName||(w.displayName="AuthorRow")}catch{}var S=o(90836);function N({authors:e}){let t=y();return(0,r.jsx)(S.Z,{children:e.slice(0,5).map((e,o)=>(0,r.jsx)(a.O,{"data-testid":"commit-stack-avatar",src:e.avatarUrl,alt:e.login??e.displayName,"data-hovercard-url":(0,n.zP)({owner:e.login??""}),square:c(e),size:t.avatarSize},`${e.login}_${o}`))})}try{N.displayName||(N.displayName="CommitAuthorStack")}catch{}function Z({author:e,repo:t,sx:o={}}){let a=y();return e?(0,r.jsx)(i.Z,{sx:{display:"flex",flexDirection:"row",alignItems:"center",...o},"data-testid":"author-link",children:e.login?(0,r.jsx)(p,{author:e,renderTooltip:a.includeTooltip,children:(0,r.jsx)(l.Z,{muted:!0,href:(0,n.OI)({repo:t,author:e.login}),"aria-label":`commits by ${e.login}`,sx:{fontWeight:a.fontWeight,whiteSpace:"nowrap",color:a.fontColor,"&:hover":{color:a.fontColor,textDecoration:"underline"}},children:e.login})}):(0,r.jsx)(s.Z,{sx:{fontWeight:a.fontWeight,whiteSpace:"nowrap",color:a.fontColor},children:e.displayName})}):null}try{Z.displayName||(Z.displayName="AuthorLink")}catch{}function I({author:e,repo:t}){return(0,r.jsx)(b,{author:e,repo:t})}function P({author:e,committer:t,repo:o}){return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(N,{authors:[e,t]}),(0,r.jsx)(Z,{author:e,repo:o,sx:{pl:1}})]})}function $({authors:e,repo:t}){return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(N,{authors:e}),e.map((o,a)=>(0,r.jsxs)(m.Fragment,{children:[(0,r.jsx)(Z,{author:o,repo:t,sx:{pl:1}}),a!==e.length-1&&(0,r.jsx)("span",{className:"pl-1",children:"and"})]},`${o.login}_${a}`))]})}function T({authors:e,repo:t}){return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(N,{authors:e}),(0,r.jsx)(k,{authors:e,repo:t})]})}function z({authors:e,committer:t,committerAttribution:o,repo:a,children:n,includeVerbs:l=!0,authorSettings:s}){let c=1===e.length&&!o,d=1===e.length&&o,p=2===e.length&&!o,u=l?{pl:1}:{};return(0,r.jsx)(i.Z,{sx:{display:"flex",flexDirection:"row",flexWrap:["wrap","wrap","wrap","nowrap"],alignItems:"center"},children:(0,r.jsxs)(g,{authorSettings:s,children:[c&&(0,r.jsx)(I,{author:e[0],repo:a}),d&&(0,r.jsx)(P,{author:e[0],committer:t,repo:a}),p&&(0,r.jsx)($,{authors:e,repo:a}),!c&&!d&&!p&&(0,r.jsx)(T,{authors:e,repo:a}),o?(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)("span",{className:"pl-1",children:l?"authored and":"and"}),(0,r.jsx)(Z,{author:t,repo:a,sx:{pl:1}}),(0,r.jsx)(i.Z,{as:"span",sx:u,children:l&&"committed"})]}):(0,r.jsx)(i.Z,{as:"span",sx:u,children:l&&"committed"}),n]})})}try{I.displayName||(I.displayName="SingleAuthor")}catch{}try{P.displayName||(P.displayName="AuthorAndCommitter")}catch{}try{$.displayName||($.displayName="TwoAuthors")}catch{}try{T.displayName||(T.displayName="MultipleAuthors")}catch{}try{z.displayName||(z.displayName="CommitAttribution")}catch{}},4751:(e,t,o)=>{o.d(t,{AF:()=>j,vC:()=>k,fQ:()=>N});var r=o(85893),a=o(85529),n=o(98833),i=o(97011),l=o(42483),s=o(88216),c=o(65379),d=o(50919),p=o(67294),u=o(52516),h=o(74121),m=o(43535),f=o(57294),x=o(73290);function g({checkRun:e}){let{icon:t,iconColor:o}=function(e){switch(e){case"check":return{icon:a.CheckIcon,iconColor:"success.fg"};case"dot-fill":return{icon:a.DotFillIcon,iconColor:"attention.fg"};case"stop":return{icon:a.StopIcon,iconColor:"muted.fg"};case"issue-reopened":return{icon:a.IssueReopenedIcon,iconColor:"muted.fg"};case"clock":return{icon:a.ClockIcon,iconColor:"attention.fg"};case"square-fill":return{icon:a.SquareFillIcon,iconColor:"fg.default"};case"skip":return{icon:a.SkipIcon,iconColor:"muted.fg"};case"alert":return{icon:a.AlertIcon,iconColor:"danger.fg"};default:return{icon:a.XIcon,iconColor:"danger.fg"}}}(e.icon),s="in_progress"===e.state;return(0,r.jsxs)(l.Z,{"data-testid":"check-run-item",sx:{display:"flex",borderBottomWidth:"1px",borderBottomStyle:"solid",borderBottomColor:"border.default",backgroundColor:"canvas.subtle",height:"38px",py:2,pr:3,pl:"12px",alignItems:"baseline"},children:[s?(0,r.jsx)(l.Z,{sx:{height:"16px",width:"16px",minWidth:"16px",alignSelf:"center",mx:"7px"},children:(0,r.jsxs)("svg",{fill:"none",viewBox:"0 0 16 16",className:"anim-rotate","aria-hidden":"true",role:"img",children:[(0,r.jsx)("path",{opacity:".5",d:"M8 15A7 7 0 108 1a7 7 0 000 14v0z",stroke:"#dbab0a",strokeWidth:"2"}),(0,r.jsx)("path",{d:"M15 8a7 7 0 01-7 7",stroke:"#dbab0a",strokeWidth:"2"}),(0,r.jsx)("path",{d:"M8 12a4 4 0 100-8 4 4 0 000 8z",fill:"#dbab0a"})]})}):(0,r.jsx)(n.Z,{icon:t,sx:{color:o,margin:"0px 7px",alignSelf:"center"}}),(0,r.jsx)(c.Z,{"aria-label":e.avatarDescription,direction:"e",children:(0,r.jsx)(x.Z,{href:e.avatarUrl,sx:{mr:2},children:(0,r.jsx)(f.O,{square:!0,src:e.avatarLogo,sx:{backgroundColor:e.avatarBackgroundColor}})})}),(0,r.jsxs)(i.Z,{sx:{overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",fontSize:"13px",color:"fg.muted"},children:[(0,r.jsxs)(i.Z,{sx:{fontWeight:"bold",color:"fg.default",mr:"2px"},children:[e.name," "]}),e.pending?(0,r.jsx)(i.Z,{sx:{fontStyle:"italic"},children:e.additionalContext}):e.additionalContext,e.description&&(0,r.jsxs)(i.Z,{children:[" ","- ",e.pending?(0,r.jsx)(i.Z,{sx:{fontStyle:"italic"},children:e.description}):e.description]})]}),(0,r.jsx)(x.Z,{href:e.targetUrl,sx:{pl:"12px",fontSize:"13px",marginLeft:"auto"},children:"Details"})]})}try{g.displayName||(g.displayName="CheckRunItem")}catch{}function y({checkRuns:e}){return(0,r.jsx)(l.Z,{sx:{display:"flex",flexDirection:"column",maxHeight:["230px","380px"],overflow:"auto"},children:e.map((e,t)=>(0,r.jsx)(g,{checkRun:e},t))})}try{y.displayName||(y.displayName="ChecksStatusBadgeFooter")}catch{}function b({checksHeaderState:e}){switch(e){case"SUCCEEDED":return(0,r.jsx)(i.Z,{sx:{fontWeight:"bold",fontSize:2},children:"All checks have passed"});case"FAILED":return(0,r.jsx)(i.Z,{sx:{color:"checks.donutError",fontWeight:"bold",fontSize:2},children:"All checks have failed"});case"PENDING":return(0,r.jsx)(i.Z,{sx:{color:"checks.donutPending",fontWeight:"bold",fontSize:2},children:"Some checks haven\u2019t completed yet"});default:return(0,r.jsx)(i.Z,{sx:{color:"checks.donutError",fontWeight:"bold",fontSize:2},children:"Some checks were not successful"})}}try{b.displayName||(b.displayName="HeaderState")}catch{}function j(e){let{combinedStatus:t,isOpen:o,onDismiss:a}=e,n=t?(0,r.jsx)(b,{checksHeaderState:t.checksHeaderState}):"Loading...";return o?(0,r.jsx)(m.V,{onClose:a,sx:{overflowY:"auto",backgroundColor:"canvas.default",boxShadow:"none",border:"1px solid",borderColor:"border.default",borderBottom:0},title:n,subtitle:t?t.checksStatusSummary:void 0,width:"xlarge",renderBody:()=>(0,r.jsx)(m.V.Body,{sx:{padding:0},children:(0,r.jsx)(u.S,{sx:{padding:0},children:t?(0,r.jsx)(y,{checkRuns:t.checkRuns}):(0,r.jsx)(l.Z,{sx:{display:"flex",justifyContent:"center",p:2},children:(0,r.jsx)(h.Z,{size:"medium"})})})})}):null}try{j.displayName||(j.displayName="CheckStatusDialog")}catch{}let v={success:{circled:a.CheckCircleIcon,filled:a.CheckCircleFillIcon,default:a.CheckIcon,color:"checks.donutSuccess"},pending:{circled:a.CircleIcon,filled:a.DotFillIcon,default:a.DotFillIcon,color:"checks.donutPending"},error:{circled:a.XCircleIcon,filled:a.XCircleFillIcon,default:a.XIcon,color:"checks.donutError"}};function C({descriptionText:e,icon:t,iconColor:o}){return(0,r.jsxs)("span",{"data-testid":"checks-status-badge-icon-only",children:[(0,r.jsx)(n.Z,{icon:t,"aria-label":"See all checks",sx:{color:o}}),e&&(0,r.jsxs)(i.Z,{children:[" ",e]})]})}function k(e){let{statusRollup:t,combinedStatus:o,variant:a="default",disablePopover:n,buttonSx:i,size:u="medium",descriptionText:h=""}=e,[m,f]=(0,p.useState)(!1),x=(0,p.useId)(),g=(0,p.useRef)(null),y=v[t],{icon:b,iconColor:k}={icon:y?.[a]||v.error[a],iconColor:y?.color||v.error.color};return n?(0,r.jsx)(C,{descriptionText:h,icon:b,iconColor:k}):(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(l.Z,{onClick:()=>{f(!0),e.onWillOpenPopup},onMouseEnter:e.onWillOpenPopup,children:h?(0,r.jsx)(s.r,{"data-testid":"checks-status-badge-button",leadingVisual:b,variant:"invisible",size:u,"aria-label":o?.checksStatusSummary??`Status checks: ${t}`,sx:{p:1,color:"fg.default",fontWeight:"normal",svg:{color:k},...i},ref:g,children:h}):(0,r.jsx)(c.Z,{id:x,"aria-label":o?.checksStatusSummary??t,direction:"se",sx:{mr:2},children:(0,r.jsx)(d.h,{"data-testid":"checks-status-badge-icon",icon:b,variant:"invisible",size:u,"aria-labelledby":x,sx:{py:0,px:0,svg:{color:k},":hover:not([disabled])":{bg:"pageHeaderBg"},...i},ref:g})})}),m&&(0,r.jsx)(j,{combinedStatus:o,isOpen:m,onDismiss:()=>{f(!1),setTimeout(()=>{g.current?.focus()},0)}})]})}try{C.displayName||(C.displayName="IconOnlyStatus")}catch{}try{k.displayName||(k.displayName="ChecksStatusBadge")}catch{}var w=o(78212),S=o(89445);function N(e,t){let[o,r]=(0,p.useState)(),[a,n]=(0,p.useState)(),i=(0,p.useCallback)(async()=>{if(a!==e&&(n(e),r(void 0),e)){let o=(0,w.S$)(t,e),a=await (0,S.v)(o);r(await a.json())}},[e,a,t]);return[o,i]}},68912:(e,t,o)=>{o.d(t,{m:()=>h,z:()=>p});var r=o(85893),a=o(39773),n=o(37169),i=o(85529),l=o(42483),s=o(50919),c=o(67294),d=o(23978);function p(e){if("clipboard"in navigator)return navigator.clipboard.writeText(e);let t=document.body;if(!t)return Promise.reject(Error());let o=function(e){let t=document.createElement("pre");return t.style.width="1px",t.style.height="1px",t.style.position="fixed",t.style.top="5px",t.textContent=e,t}(e);return t.appendChild(o),!function(e){if("clipboard"in navigator)return navigator.clipboard.writeText(e.textContent||"");let t=getSelection();if(null==t)return Promise.reject(Error());t.removeAllRanges();let o=document.createRange();o.selectNodeContents(e),t.addRange(o),document.execCommand("copy"),t.removeAllRanges(),Promise.resolve()}(o),t.removeChild(o),Promise.resolve()}let u=({sx:e})=>(0,r.jsx)(l.Z,{"aria-label":"Copied!",sx:{display:"inline-block",color:"success.fg",p:1,mr:1,...e},children:(0,r.jsx)(i.CheckIcon,{})});function h({icon:e=i.CopyIcon,size:t="medium",onCopy:o,sx:a,textToCopy:l,tooltipProps:h,confirmationComponent:f=(0,r.jsx)(u,{sx:a}),ariaLabel:x,accessibleButton:g,hasPortalTooltip:y=!1}){let[b,j]=c.useState(!1),v=(0,n.Z)(),C=()=>{j(!0),p(l),o?.(),setTimeout(()=>v()&&j(!1),2e3)},k=x??`Copy ${l} to clipboard`;return y?(0,r.jsx)(m,{label:k,textToCopy:l,copied:b,onClickCopy:C,confirmationComponent:f,tooltipProps:h,sx:{...a}}):b?(0,r.jsx)(r.Fragment,{children:f}):(0,r.jsx)(d.u,{text:k,"aria-label":k,...h,sx:{position:"absolute"},children:(0,r.jsx)(s.h,{"aria-label":k,icon:e,variant:"invisible",size:t,tabIndex:!1===g?-1:0,sx:{...a},onClick:C})})}function m({icon:e=i.CopyIcon,size:t="medium",label:o,accessibleButton:n,copied:d,onClickCopy:p,tooltipProps:u,sx:h}){let m=c.useRef(null),[f,x]=(0,a.g)({contentRef:m,"aria-label":d?"Copied!":o,...u});return(0,r.jsxs)(l.Z,{ref:m,...f,children:[d?(0,r.jsx)(l.Z,{sx:{...h},children:(0,r.jsx)(l.Z,{as:"span",sx:{display:"inline-block",color:"success.fg",p:1,mr:1},children:(0,r.jsx)(i.CheckIcon,{})})}):(0,r.jsx)(s.h,{"aria-label":o,icon:e,variant:"invisible",size:t,tabIndex:!1===n?-1:0,sx:{...h},onClick:p}),x]})}try{u.displayName||(u.displayName="CopyConfirmationCheck")}catch{}try{h.displayName||(h.displayName="CopyToClipboardButton")}catch{}try{m.displayName||(m.displayName="PortalTooltipCopyButton")}catch{}},95628:(e,t,o)=>{o.d(t,{M:()=>n});let r=e=>{let t=getComputedStyle(e,null);return["overflow","overflow-y","overflow-x"].some(e=>{let o=t.getPropertyValue(e);return"auto"===o||"scroll"===o})},a=(e,t)=>e&&null!==e.parentNode?a(e.parentNode,t.concat([e])):t;function n(e){if(!(e instanceof HTMLElement||e instanceof SVGElement))return;let t=a(e.parentNode,[]);for(let e of t)if((e instanceof HTMLElement||e instanceof SVGElement)&&r(e))return e;return document.scrollingElement||document.documentElement}},78806:(e,t,o)=>{o.d(t,{Z:()=>r});let r=(e,t)=>{let o=new URL(e,window.location.origin),r=new URL(t,window.location.origin),a=r.href.includes("#");return a&&o.host===r.host&&o.pathname===r.pathname&&o.search===r.search}},2048:(e,t,o)=>{o.d(t,{g:()=>a,y:()=>n});var r=o(17891);let a=()=>r.M()?.enabled_features??{},n=e=>!!a()[e]},41155:(e,t,o)=>{o.d(t,{a:()=>a});var r=o(67294);function a({focusRowRef:e,mouseRowRef:t}){let[o,a]=r.useState(!1);return r.useEffect(()=>{if(e.current&&t.current){let o=()=>{let t=e.current?.querySelector(".PRIVATE_TreeView-item-content-text");t?.scrollWidth!==t?.offsetWidth&&a(!0)};e.current.onfocus=()=>{o()},e.current.onblur=()=>{a(!1)},t.current.onmouseenter=()=>{o()},t.current.onmouseleave=()=>{a(!1)}}},[e,t]),o}},68203:(e,t,o)=>{o.d(t,{s:()=>c});var r=o(67294),a=o(89250),n=o(12599),i=o(78806),l=o(45055),s=o(15981);let c=()=>{let{routes:e,history:t}=r.useContext(l.I),c=(0,a.s0)();return r.useCallback((r,a)=>{let l=(0,n.i3)(r).pathname,d=!(0,n.fp)(e,l);if(d){let e=t.createHref(r);(async()=>{let{softNavigate:t}=await Promise.all([o.e("vendors-node_modules_github_turbo_dist_turbo_es2017-esm_js"),o.e("ui_packages_soft-navigate_soft-navigate_ts")]).then(o.bind(o,75198));t(e)})()}else{(0,i.Z)(location.href,r.toString())||(0,s.LD)("react"),c(r,a);let{turbo:e,...t}=window.history.state;window.history.replaceState({...t,skipTurbo:!0},"",location.href)}},[t,c,e])}},32769:(e,t,o)=>{o.d(t,{H:()=>l,d:()=>i});var r=o(85893),a=o(67294);let n=a.createContext({});function i({repository:e,children:t}){return(0,r.jsxs)(n.Provider,{value:e,children:[" ",t," "]})}function l(){return a.useContext(n)}try{n.displayName||(n.displayName="CurrentRepositoryContext")}catch{}try{i.displayName||(i.displayName="CurrentRepositoryProvider")}catch{}},57294:(e,t,o)=>{o.d(t,{O:()=>l});var r=o(85893),a=o(67294),n=o(26012),i=o(86283);let l=(0,a.forwardRef)(function({src:e,size:t=20,...o},l){let s=(0,a.useMemo)(()=>{let o=new URL(e,i.ssrSafeLocation.origin);return o.searchParams.has("size")||o.searchParams.has("s")||o.searchParams.set("size",String(2*Number(t))),o.toString()},[e,t]);return(0,r.jsx)(n.Z,{ref:l,src:s,size:t,"data-testid":"github-avatar",...o})});try{l.displayName||(l.displayName="GitHubAvatar")}catch{}},45222:(e,t,o)=>{o.d(t,{h:()=>m});var r=o(85893),a=o(42379),n=o(15173),i=o(41905),l=o(86010),s=o(67294),c=o(15388);let d=c.ZP.span.withConfig({displayName:"ControlledTooltip__TooltipBase",componentId:"sc-a342ffc-0"})`
  &::before {
    position: absolute;
    z-index: 1000001;
    display: none;
    width: 0px;
    height: 0px;
    color: ${(0,a.U2)("colors.neutral.emphasisPlus")};
    pointer-events: none;
    content: '';
    border: 6px solid transparent;
    opacity: 0;
  }
  &::after {
    position: absolute;
    z-index: 1000000;
    display: none;
    padding: 0.5em 0.75em;
    font: normal normal 11px/1.5 ${(0,a.U2)("fonts.normal")};
    -webkit-font-smoothing: subpixel-antialiased;
    color: ${(0,a.U2)("colors.fg.onEmphasis")};
    text-align: center;
    text-decoration: none;
    text-shadow: none;
    text-transform: none;
    letter-spacing: normal;
    word-wrap: break-word;
    white-space: pre;
    pointer-events: none;
    content: attr(aria-label);
    background: ${(0,a.U2)("colors.neutral.emphasisPlus")};
    border-radius: ${(0,a.U2)("radii.1")};
    opacity: 0;
  }
  /* delay animation for tooltip */
  @keyframes tooltip-appear {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }
  &.tooltipped-open,
  &:hover,
  &:active,
  &:focus {
    &::before,
    &::after {
      display: inline-block;
      text-decoration: none;
      animation-name: tooltip-appear;
      animation-duration: 0.1s;
      animation-fill-mode: forwards;
      animation-timing-function: ease-in;
      animation-delay: 0.4s;
    }
  }

  &.tooltipped-no-delay.tooltipped-open,
  &.tooltipped-no-delay:hover,
  &.tooltipped-no-delay:active,
  &.tooltipped-no-delay:focus {
    &::before,
    &::after {
      animation-delay: 0s;
    }
  }

  /* Tooltipped south */
  &.tooltipped-s,
  &.tooltipped-se,
  &.tooltipped-sw {
    &::after {
      top: 100%;
      right: 50%;
      margin-top: 6px;
    }
    &::before {
      top: auto;
      right: 50%;
      bottom: -7px;
      margin-right: -6px;
      border-bottom-color: ${(0,a.U2)("colors.neutral.emphasisPlus")};
    }
  }
  &.tooltipped-se {
    &::after {
      right: auto;
      left: 50%;
      margin-left: -${(0,a.U2)("space.3")};
    }
  }
  &.tooltipped-sw::after {
    margin-right: -${(0,a.U2)("space.3")};
  }
  /* Tooltips above the object */
  &.tooltipped-n,
  &.tooltipped-ne,
  &.tooltipped-nw {
    &::after {
      right: 50%;
      bottom: 100%;
      margin-bottom: 6px;
    }
    &::before {
      top: -7px;
      right: 50%;
      bottom: auto;
      margin-right: -6px;
      border-top-color: ${(0,a.U2)("colors.neutral.emphasisPlus")};
    }
  }
  &.tooltipped-ne {
    &::after {
      right: auto;
      left: 50%;
      margin-left: -${(0,a.U2)("space.3")};
    }
  }
  &.tooltipped-nw::after {
    margin-right: -${(0,a.U2)("space.3")};
  }
  /* Move the tooltip body to the center of the object. */
  &.tooltipped-s::after,
  &.tooltipped-n::after {
    transform: translateX(50%);
  }
  /* Tooltipped to the left */
  &.tooltipped-w {
    &::after {
      right: 100%;
      bottom: 50%;
      margin-right: 6px;
      transform: translateY(50%);
    }
    &::before {
      top: 50%;
      bottom: 50%;
      left: -7px;
      margin-top: -6px;
      border-left-color: ${(0,a.U2)("colors.neutral.emphasisPlus")};
    }
  }
  /* tooltipped to the right */
  &.tooltipped-e {
    &::after {
      bottom: 50%;
      left: 100%;
      margin-left: 6px;
      transform: translateY(50%);
    }
    &::before {
      top: 50%;
      right: -7px;
      bottom: 50%;
      margin-top: -6px;
      border-right-color: ${(0,a.U2)("colors.neutral.emphasisPlus")};
    }
  }
  &.tooltipped-align-right-2::after {
    right: 0;
    margin-right: 0;
  }
  &.tooltipped-align-right-2::before {
    right: 15px;
  }
  &.tooltipped-align-left-2::after {
    left: 0;
    margin-left: 0;
  }
  &.tooltipped-align-left-2::before {
    left: 10px;
  }
  ${n.Z};
`,p=()=>()=>void 0,u=()=>!1,h=()=>!0,m=(0,s.forwardRef)(function({direction:e="n",className:t,text:o,noDelay:a,align:n,wrap:c,open:m=!1,portalProps:f={},...x},g){let y=(0,s.useSyncExternalStore)(p,h,u),b=(0,l.W)(t,`tooltipped-${e}`,n&&`tooltipped-align-${n}-2`,a&&"tooltipped-no-delay",c&&"tooltipped-multiline",m&&"tooltipped-open");return y?(0,r.jsx)(i.h,{...f,children:(0,r.jsx)(d,{ref:g,role:"tooltip","aria-label":o,...x,sx:{position:"fixed",zIndex:1,...x.sx},className:b})}):null});try{m.displayName||(m.displayName="ControlledTooltip")}catch{}},52793:(e,t,o)=>{o.d(t,{u:()=>s});var r=o(85893),a=o(48030),n=o(67294),i=o(45222),l=o(95628);let s=(0,n.forwardRef)(function({contentRef:e,open:t,anchoredPositionAlignment:o,anchorSide:s,anchorOffset:c,alignmentOffset:d,allowOutOfBounds:p,...u},h){let m=(0,n.useRef)(null);(0,n.useImperativeHandle)(h,()=>m.current);let f=(0,n.useRef)({left:0,top:0}),x=(0,n.useSyncExternalStore)((0,n.useCallback)(o=>{if(!m.current||!e.current||!t)return()=>void 0;let r=(0,l.M)(e.current);return r?.addEventListener("scroll",o),()=>{r?.removeEventListener("scroll",o)}},[e,t]),(0,n.useCallback)(()=>{if(!m.current||!e.current)return f.current;let t=(0,a.N)(m.current,e.current,{align:o??"center",side:s??"outside-top",alignmentOffset:d??0,anchorOffset:c??0,allowOutOfBounds:p});return(t.left!==f.current.left||t.top!==f.current.top)&&(f.current=t),f.current},[e,d,c,o,s,p]),(0,n.useCallback)(()=>f.current,[]));return(0,r.jsx)(i.h,{...u,ref:m,open:t,style:{position:"absolute",...x,...u.style}})});try{s.displayName||(s.displayName="PortalTooltip")}catch{}},39773:(e,t,o)=>{o.d(t,{g:()=>i});var r=o(85893),a=o(67294),n=o(52793);function i({includeAriaAttributes:e=!0,...t}){let o=(0,a.useId)(),[i,l]=(0,a.useState)(!1),s=t.open??i,c=t.id??o;return[{...e?{"aria-describedby":c}:{},onPointerEnter:(0,a.useCallback)(()=>l(!0),[]),onPointerLeave:(0,a.useCallback)(()=>l(!1),[])},(0,r.jsx)(n.u,{...t,open:s,id:c},"portal-tooltip")]}},60348:(e,t,o)=>{o.d(t,{r:()=>c});var r=o(85893),a=o(67294),n=o(12599),i=o(79655),l=o(45055),s=o(86283);let c=a.forwardRef(({to:e,reloadDocument:t,...o},c)=>{let{routes:d}=a.useContext(l.I),p=(0,n.i3)(e,s.ssrSafeLocation.pathname).pathname;return t=t??!(0,n.fp)(d,p),(0,r.jsx)(i.rU,{to:e,...o,reloadDocument:t,ref:c})});c.displayName="Link"}}]);
//# sourceMappingURL=ui_packages_commit-attribution_index_ts-ui_packages_commit-checks-status_index_ts-ui_packages-5acf59-07b10888ceb7.js.map