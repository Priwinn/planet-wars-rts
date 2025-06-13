package games.planetwars.runners

import games.planetwars.agents.PlanetWarsAgent
import games.planetwars.agents.random.PureRandomAgent
import games.planetwars.core.*
import games.planetwars.agents.RemoteAgent
import py4j.GatewayServer

class GameRunnerPy4JWrapper(
    val agent1: PlanetWarsAgent,
    val agent2: PlanetWarsAgent,
    val gameParams: GameParams,
) {
    private val gameRunner = GameRunnerCoRoutines(
        agent1 = agent1,
        agent2 = agent2,
        gameParams = gameParams,
        timeoutMillis = 500000 // No timeout for Py4J, as it will block until the agent responds
    )
    
    fun newGame() {
        gameRunner.newGame()
    }

    fun stepGame(): Map<String, Any> {
        val forwardModel = gameRunner.stepGame()
        return mapOf(
            "isTerminal" to forwardModel.isTerminal(),
            "tick" to forwardModel.state.gameTick,
            "planets" to forwardModel.state.planets.map { planet ->
                mapOf(
                    "x" to planet.position.x,
                    "y" to planet.position.y,
                    "owner" to (planet.owner?.ordinal ?: -1),
                    "numShips" to planet.nShips,
                    "growthRate" to planet.growthRate,
                    "id" to planet.id,
                    "radius" to planet.radius,
                    "transporter" to planet.transporter?.let { transporter ->
                        mapOf(
                            "owner" to transporter.owner.ordinal,
                            "numShips" to transporter.nShips,
                            "sourceIndex" to transporter.sourceIndex,
                            "destinationIndex" to transporter.destinationIndex,
                            "positionX" to transporter.s.x,
                            "positionY" to transporter.s.y,
                            "velocityX" to transporter.v.x,
                            "velocityY" to transporter.v.y
                        )
                    }
                )
            },
            "leader" to forwardModel.getLeader().ordinal,
            "statusString" to forwardModel.statusString()
        )
    }

    fun getGameState(): Map<String, Any> {
        return mapOf(
            "isTerminal" to gameRunner.forwardModel.isTerminal(),
            "tick" to gameRunner.forwardModel.state.gameTick,
            "planets" to gameRunner.forwardModel.state.planets.map { planet ->
                mapOf(
                    "x" to planet.position.x,
                    "y" to planet.position.y,
                    "owner" to (planet.owner?.ordinal ?: -1),
                    "numShips" to planet.nShips,
                    "growthRate" to planet.growthRate,
                    "id" to planet.id,
                    "radius" to planet.radius,
                    "transporter" to planet.transporter?.let { transporter ->
                        mapOf(
                            "owner" to transporter.owner.ordinal,
                            "numShips" to transporter.nShips,
                            "sourceIndex" to transporter.sourceIndex,
                            "destinationIndex" to transporter.destinationIndex,
                            "positionX" to transporter.s.x,
                            "positionY" to transporter.s.y,
                            "velocityX" to transporter.v.x,
                            "velocityY" to transporter.v.y
                        )
                    }


                )
            }
        )
    }
}

class Py4JEntryPoint {
    fun createGameRunner(): GameRunnerPy4JWrapper {
        val gameParams = GameParams(numPlanets = 20, maxTicks = 500)
        val agent1 = PureRandomAgent()
        val agent2 = RemoteAgent("<specified by remote server>", port = 8080)
        return GameRunnerPy4JWrapper(agent1, agent2, gameParams)
    }
}


fun main() {
    val entryPoint = Py4JEntryPoint()
    val gameRunner = entryPoint.createGameRunner()
    val gatewayServer = GatewayServer(entryPoint)
    gatewayServer.start()
    println("Gateway Server Started") 
}