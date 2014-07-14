<?php session_start(); 

if($_SESSION['logedin'] == false)
  {
    header('Location: http://ee-ug1.ee.ic.ac.uk/actual_web2/');
  }

?>

<!DOCTYPE html>
<!--[if IE 8]>    <html class="no-js lt-ie9" lang="en"> <![endif]-->
<!--[if gt IE 8]><!--> <html class="no-js" lang="en"> <!--<![endif]-->
<head>
  <meta charset="utf-8" />
 <link rel="shortcut icon" href="http://ee-ug1.ee.ic.ac.uk/actual_web2/images/favicon.ico"/>
  <!-- Set the viewport width to device width for mobile -->
  <meta name="viewport" content="width=device-width" />

  <title>Welcome to your AutoHome!</title>

  <link rel="stylesheet" href="css/normalize.css">
  </br>
  <link rel="stylesheet" href="css/foundation.css">

  <script src="custom.modernizr.js"></script>
  <script src="//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min.js"></script>
</head>
<body>

<!-- Nav and Banner -->
  
  <div class="row">
    <div class="large-12 columns">
      <ul class="button-group">
        <li><a href="home.php" class="button" style="width: 200px">Home Page</a></li>
        <li><a href="control.php" class="button" style="width: 200px">House Control</a></li>
        <li><a href="stats.php" class="button" style="width: 200px">Statistics</a></li>
        <li><a href="thermostat.php" class="button" style="width: 200px">Thermostat Setting</a></li>
      </ul>
      
      <!-- Main Banner Image - place in a paragraph to enforce the modular scale spacing -->
      <p><img src="images/LogoTitle.png" /></p>
	  <img id="img" src="images/dance.gif" style="width: 800px; height: 800px" onclick="yesno('input-filter-height[1]');"/>
   
		<footer class="row" style="width: 830px">
    <div class="large-12 columns" style="width:800px">
      <hr />
      <div class="row">
        <div class="large-6 columns"  style="width:800px">
         <p style="text-align:right"><font color="#ffffff">&copy; Copyright @AutoHome</font>
           <a href="logout.php">Log out</a></p>
        </div>
      </div>
    </div> 
  </footer>
   </div>
  </div>

  <!-- End Nav and Banner -->
 

			<center>
			<input type="checkbox" class="checkit" style="display:none;" name="input-filter-height[1]" id="input-filter-height[1]" value="1";/>
			
			
			
			</center>
	
	<script type='text/javascript'>	
		var ws = new WebSocket("ws://ec2-54-214-164-65.us-west-2.compute.amazonaws.com:8888/ws");
		
		function yesno(thecheckbox) {
		
			var checkboxvar = document.getElementById(thecheckbox);
			if (!checkboxvar.checked) {
				ws.send('{"type":"light_control", "light_ID": 1, "status": 0}');
				document.getElementById("img").src="images/gerbil.gif";   
			}
			else {
				ws.send('{"type":"light_control", "light_ID": 1, "status": 1}');
				document.getElementById("img").src="images/tophat.gif";
			}
			checkboxvar.checked= !checkboxvar.checked;
		}
	
			
			ws.onopen = function () {
				ws.send('{"type":"light_request", "light_ID": 1}');
			};
			
			ws.onmessage = function (evt) {
			    console.log("message: " + evt.data);
				msg_data = jQuery.parseJSON(evt.data);
				console.log("got ws status: " + msg_data.status);
				//if(msg_data.type=="light_request_response")
				{
				if(msg_data.status==0)
				{
					document.getElementById('input-filter-height[1]').checked=true;
					document.getElementById("img").src="images/gerbil.gif";
				}
				else
				{
					document.getElementById('input-filter-height[1]').checked=false;
					document.getElementById("img").src="images/tophat.gif";
				}}
			};
	</script>
  
  
  <!-- Footer -->


 

</body>
</html>