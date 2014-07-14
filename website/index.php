<?php session_start(); 
//$_SESSION['logedin'] = false;
//$_SESSION['user'] = "";
if(@$_SESSION['logedin'] == true)
{
  header('Location: http://ee-ug1.ee.ic.ac.uk/actual_web2/home.php');
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

  <title>Please Log In!</title>

  <link rel="stylesheet" href="css/normalize.css">
  <link rel="stylesheet" href="css/foundation.css">
  <link rel="stylesheet" href="css/main.css">

  <script src="js/vendor/custom.modernizr.js"></script>
  <script src="//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min.js"></script>

</head>
<body>

<!-- Nav and Banner -->
<div class="row content">
  <div class="row">
    <div class="large-12 columns">
      <p><img src="images/LogoTitle.png" /></p>
    </div>
  </div>

  <!--<div class="row no-padding">
    put links here
  <div>!-->

  <div class="row">
  
    <div class="large-8 columns">
      <h4>Please Log in!</h4>

      <form action="" method="post" style="max-width: 400px" id="form">
        <p>Username: <input type="text" name="username" id="username" placeholder="Enter your username"></p>
        <p>Password: <input type="password" name="password" id="password" placeholder="Enter your password"></p>
        <input type="submit"  class="radius button" value="Log In">
      </form>
    </div>    
  </div>
</div>
  
  

  
  <!-- Footer -->
<footer>
  <div class="row">
    <div class="large-12 columns">
      <p class="text-center">&copy; 2013 AutoHome</p>
    </div> 
</div>
</footer>


  <script type="text/javascript">

   $("#form").submit(function(e){
    e.preventDefault();
    var str = $("#form").serialize();
    $.post('login.php', str, function(result){
      if(result=="False")
      {
        alert("incorrect username or password");
      }
      else
      {
        window.location.replace("home.php");
      }
      })
    });

  </script>
 

</body>
</html>