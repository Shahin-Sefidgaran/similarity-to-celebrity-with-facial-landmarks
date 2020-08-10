//DOM
const $ = document.querySelector.bind(document);

//APP
let App = {};
function upload(files) {
  var that = this;
  var formData = new FormData();

  // add assoc key values, this will be posts values
   formData.append("name", files[0].name );
   formData.append("photo", files[0]);

  // $.ajax({
  //     type: "POST",
  //     url: "http://localhost:8000/upload/",
  //     xhr: function () {
  //         var myXhr = $.ajaxSettings.xhr();
  //         if (myXhr.upload) {
  //             myXhr.upload.addEventListener('progress', that.progressHandling, false);
  //         }
  //         return myXhr;
  //     },
  //     success: function (data) {
  //         // your callback here

  //     }})
  //   }

  var requestOptions = {
    method: 'POST',
    body: formData,
    redirect: 'follow'
  };
  
  fetch("http://88.198.21.151/upload/", requestOptions)
    .then(response => response.text())
    .then(result =>{
      result = JSON.parse(result);
      result = result.result_photo
      result = result.split(',')
      
      let theImg1 = $("#resultImg1");
      let theImg2 = $("#resultImg2");
      let theImg3 = $("#resultImg3");

      theImg1.src = result[0].split(' ')[0]
      theImg2.src = result[1].split(' ')[1]
      theImg3.src = result[2].split(' ')[1]
    })
    .catch(error => console.log('error', error));

}

App.init = function () {
  //Init
  function handleFileSelect(evt) {
    const files = evt.target.files; // FileList object
    console.log(files);
    //files template
    let template = `${Object.keys(files).
    map(file => `<div class="file file--${file}">
     <div class="name"><span>${files[file].name}</span></div>
     <div class="progress active"></div>
     <div class="done">
	<a href="" target="_blank">
      <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" x="0px" y="0px" viewBox="0 0 1000 1000">
		<g><path id="path" d="M500,10C229.4,10,10,229.4,10,500c0,270.6,219.4,490,490,490c270.6,0,490-219.4,490-490C990,229.4,770.6,10,500,10z M500,967.7C241.7,967.7,32.3,758.3,32.3,500C32.3,241.7,241.7,32.3,500,32.3c258.3,0,467.7,209.4,467.7,467.7C967.7,758.3,758.3,967.7,500,967.7z M748.4,325L448,623.1L301.6,477.9c-4.4-4.3-11.4-4.3-15.8,0c-4.4,4.3-4.4,11.3,0,15.6l151.2,150c0.5,1.3,1.4,2.6,2.5,3.7c4.4,4.3,11.4,4.3,15.8,0l308.9-306.5c4.4-4.3,4.4-11.3,0-15.6C759.8,320.7,752.7,320.7,748.4,325z"</g>
		</svg>
						</a>
     </div>
    </div>`).
    join("")}`;
  
    $("#drop").classList.add("hidden");
    $("footer").classList.add("hasFiles");
    $(".importar").classList.add("active");
    setTimeout(() => {
      $(".list-files").innerHTML = template;
    }, 1000);


    upload(files)
    Object.keys(files).forEach(file => {
      let load = 2000 + file * 2000; // fake load
      setTimeout(() => {
        $(`.file--${file}`).querySelector(".progress").classList.remove("active");
        $(`.file--${file}`).querySelector(".done").classList.add("anim");
      }, load);
    });
  }

  // trigger input
  $("#triggerFile").addEventListener("click", evt => {
    evt.preventDefault();
    $("input[type=file]").click();
  });

  // drop events
  $("#drop").ondragleave = evt => {
    $("#drop").classList.remove("active");
    evt.preventDefault();
  };
  $("#drop").ondragover = $("#drop").ondragenter = evt => {
    $("#drop").classList.add("active");
    evt.preventDefault();
  };
  $("#drop").ondrop = evt => {
    $("input[type=file]").files = evt.dataTransfer.files;
    $("footer").classList.add("hasFiles");
    $("#drop").classList.remove("active");
    evt.preventDefault();
  };

  //upload more
  $(".importar").addEventListener("click", () => {
    $(".list-files").innerHTML = "";
    $("footer").classList.remove("hasFiles");
    $(".importar").classList.remove("active");
    setTimeout(() => {
      $("#drop").classList.remove("hidden");
    }, 500);
  });

  // input change
  $("input[type=file]").addEventListener("change", handleFileSelect);
}();