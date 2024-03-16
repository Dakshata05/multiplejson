// function goBack() {
//     window.location.href = '../page.html';
// }

// function openWindow(videoSrc) {
//     // Get the main container
//     var mainContainer = document.getElementById('main-container');

//     // Remove any previously added videos
//     var existingVideos = mainContainer.querySelectorAll('video');
//     existingVideos.forEach(function(video) {
//         video.remove();
//     });

//     // Create a new video element
//     var videoElement = document.createElement('video');
//     videoElement.src = videoSrc;
//     videoElement.controls = true;
//     videoElement.autoplay=true;
//     videoElement.loop=true;
//     videoElement.style.width = '500px'; // Set width of the video (for example)
//     videoElement.style.height = '300px'; // Set height of the video (for example)

//     // Append the video element to the main container
//     mainContainer.appendChild(videoElement);}
//     function toggleVideo() {
//         // Get the main container
//         var mainContainer = document.getElementById('main-container');

//         // Check if any video is present
//         var existingVideos = mainContainer.querySelectorAll('video');
//         if (existingVideos.length > 0) {
//             // If a video is present, remove it
//             existingVideos.forEach(function(video) {
//                 video.remove();
//             });
//         } else {
//             // If no video is present, do nothing
//         }
//     }