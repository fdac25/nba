const TEAM_TO_ABBREVIATION = {
  "Atlanta Hawks": "ATL",
  "Boston Celtics": "BOS",
  "Brooklyn Nets": "BRK",
  "Charlotte Hornets": "CHO",
  "Chicago Bulls": "CHI",
  "Cleveland Cavaliers": "CLE",
  "Dallas Mavericks": "DAL",
  "Denver Nuggets": "DEN",
  "Detroit Pistons": "DET",
  "Golden State Warriors": "GSW",
  "Houston Rockets": "HOU",
  "Indiana Pacers": "IND",
  "Los Angeles Clippers": "LAC",
  "Los Angeles Lakers": "LAL",
  "Memphis Grizzlies": "MEM",
  "Miami Heat": "MIA",
  "Milwaukee Bucks": "MIL",
  "Minnesota Timberwolves": "MIN",
  "New Orleans Pelicans": "NOP",
  "New York Knicks": "NYK",
  "Oklahoma City Thunder": "OKC",
  "Orlando Magic": "ORL",
  "Philadelphia 76ers": "PHI",
  "Phoenix Suns": "PHO",
  "Portland Trail Blazers": "POR",
  "Sacramento Kings": "SAC",
  "San Antonio Spurs": "SAS",
  "Toronto Raptors": "TOR",
  "Utah Jazz": "UTA",
  "Washington Wizards": "WAS",
};

const ABBREVIATION_TO_TEAM = Object.fromEntries(
  Object.entries(TEAM_TO_ABBREVIATION).map(([team, abbreviation]) => [
    abbreviation,
    team,
  ])
);

const submitButton = document.getElementById("submit-button");

submitButton.addEventListener("click", () => {
  const homeTeam = document.getElementById("home-team");
  const awayTeam = document.getElementById("away-team");

  if (homeTeam.value === awayTeam.value) {
    alert("Please select two different teams");
    return;
  }

  const homeAbbreviation = TEAM_TO_ABBREVIATION[homeTeam.value];
  const awayAbbreviation = TEAM_TO_ABBREVIATION[awayTeam.value];

  fetch("/predict", {
    method: "POST",
    body: JSON.stringify({
      homeTeam: homeAbbreviation,
      awayTeam: awayAbbreviation,
    }),
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("result").textContent = `The winner is ${
        ABBREVIATION_TO_TEAM[data.winner]
      }`;
    });
});
