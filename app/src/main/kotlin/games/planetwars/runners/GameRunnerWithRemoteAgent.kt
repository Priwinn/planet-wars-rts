package games.planetwars.runners


import games.planetwars.agents.Action
import games.planetwars.agents.DoNothingAgent
import games.planetwars.agents.PlanetWarsAgent
import games.planetwars.agents.RemoteAgent
import games.planetwars.agents.random.PureRandomAgent
import games.planetwars.agents.random.SlowRandomAgent
import games.planetwars.core.*
import kotlinx.coroutines.*
import kotlin.system.measureTimeMillis


fun main() {
    val gameParams = GameParams(numPlanets = 20, maxTicks = 500)
//    val agent1 = DoNothingAgent()
    val agent1 = PureRandomAgent()
    val agent2 = RemoteAgent("<specified by remote server>", port = 8080)
    val gameRunner = GameRunnerCoRoutines(agent1, agent2, gameParams, timeoutMillis = 10)
    val finalModel = gameRunner.runGame()
    println("Game over!")
    println(finalModel.statusString())

    // time to run a bunch of games
    val nGames = 3
    val results = gameRunner.runGames(nGames)
    println(results)
    print(agent2.getAgentType())
}
