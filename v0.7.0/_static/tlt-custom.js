/* Extra tlt-specific javascript */

$(document).ready(function(){

   /* open external links in a new tab */
   $('a[class*=external]').attr({target: '_blank', rel: 'noopener'});

   /* add word break points (zero-width space) after a period in really long titles */
   $('h1').html(function(index, html){
     return html.replace(/\./g, '.\u200B');
   });

   /* copy image alt tags as title so hover text tool tip by browser
    * (Looks like the myst-parser isn't passing the title tag through to Sphinx,
    * but is passing the alt tag) */
   $("img[alt]").each(function(){
       $(this).attr('title', $(this).attr('alt'));
   });
});
