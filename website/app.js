// get the submit button
const submitButton = document.getElementById("submit-button");

// add an event listener to the submit button
submitButton.addEventListener("click", () => {
  // get the team1 and team2 select elements
  const team1 = document.getElementById("team1");
  const team2 = document.getElementById("team2");

  // get the value of the team1 and team2 select elements
  const team1Value = team1.value;
  const team2Value = team2.value;

  console.log(team1Value, team2Value);
});
