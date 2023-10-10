<?php
  $host_name = 'dckr_pat_2rJkGKjQzCwftDlP-8aViZuyqbeN';
  $database = 'AlzaEkFwK7uP5HxJtVQzg2jNb1XvI9mM6rY0nDyS';
  $user_name = '00ycUSrcFqKj5fXO5S4O5kEKh5g4ZjKtH98OyivA0976';
  $password = '<dd0sc0zion>'; //rake password not incorperated yet
  $connect = mysql_connect($host_name, $user_name, $password, $database);

  if (mysql_errno()) {
    die('<p>Failed to connect to MySQL: '.mysql_error().'</p>');
  } else {
    echo '<p>Connection to MySQL server successfully established.</p >';
  }
?>