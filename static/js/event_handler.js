document.addEventListener('DOMContentLoaded', domReady);

        function domReady() {
            new Dics({
                container: document.querySelectorAll('.b-dics')[0],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[1],
                hideTexts: false,
                textPosition: "bottom"

            });
        }

        function objectSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[0]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 3
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                let parts = image.src.split('/').slice(0, -1);
                switch (idx) {
                    case 0:
                        parts[parts.length-2] = 'bicycle'
                        image.src = parts.join('/')
                        break;
                    case 1:
                        parts[parts.length-2] = 'kitchen'
                        image.src = parts.join('/')
                        break;
                    case 2:
                        image.src = 'static/images/stump/';
                        break;
                    case 3:
                        image.src = 'static/images/garden/';
                        break;
                    case 4:
                        image.src = 'static/images/truck/';
                        break;   
                }
                switch (i) {
                    case 0:
                        image.src = image.src + '/gs.png';
                        break;
                    case 1:
                        image.src = image.src + '/ours.png';
                        break;
                    case 2:
                        image.src = image.src + '/sugar.png';
                        break;

                }
            }

            let scene_list = document.getElementById("object-scale-recon").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }
        function rgbNormalEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[0]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 3
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                let parts = image.src.split('/');
                switch (idx) {
                    case 0:
                        parts[parts.length-2] = 'rgb'
                        image.src = parts.join('/')
                        break;
                    case 1:
                        parts[parts.length-2] = 'normal'
                        image.src = parts.join('/')
                        break;
                }
            }

            let scene_list = document.getElementById("object-rgb-normal").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }
