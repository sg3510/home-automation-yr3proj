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

  <script src="js/vendor/custom.modernizr.js"></script>

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
    </div>
  </div>
  
  <!-- End Nav and Banner -->
  
  
  <!-- Main Content Section -->
  
  <div class="row">
  
    <div class="large-8 columns">
      <h4>Welcome, <?php echo $_SESSION['user']; ?>!</h4>
      <p><font color="#ffffff">Welcome to your AutoHome! We are here to help you save money and the enviroment, let us help you save polar bears. Using this service you can manually adjust the current setting of your thermostat, manually control appliances in your home and recieve feedback statistics on your energy usage adn expenditure.</font></p>
    </div>

        
  </div>
  
  
  <!-- Call to Action Panel -->
  <div class="row">
    <div class="large-12 columns" style="width:800px">
    
      <div class="panel">
        <h4>Do you wish to manually adjust your thermostat?</h4>
            
        <div class="row">
          <div class="large-9 columns">
            <p>Please use the following link:</p>
          </div>
          <div class="large-3 columns">
            <a href="thermostat.php" class="radius button right">Thermostat Setting</a>
          </div>
        </div>
      </div>
      <footer class="row" style="width: 830px">
    <div class="large-12 columns" style="width:800px">
      <hr />
      <div class="row">
        <div class="large-6 columns" style="width:800px">
          <p style="text-align:right"><font color="#ffffff">&copy; Copyright @AutoHome</font>
           <a href="logout.php">Log out</a></p>
        </div>
      </div>
    </div> 
  </footer>
    </div>
  </div>
  
  <!-- Footer -->
  
 


 

</body>
</html>