package games.planetwars.agents


import games.planetwars.agents.Action
import games.planetwars.agents.PlanetWarsAgent
import games.planetwars.agents.random.PureRandomAgent
import games.planetwars.core.*
import games.planetwars.runners.GameRunnerCoRoutines
import io.ktor.client.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.websocket.*
import io.ktor.websocket.*
import json_rmi.*
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject

fun main() {
    val agent1 = RemoteAgent("games.planetwars.agents.random.CarefulRandomAgent")
    val agent2 = PureRandomAgent()
    val gameParams = GameParams(
        numPlanets = 4,
        initialNeutralRatio = 0.0,
        maxTicks = 20,
    )
    val runner = GameRunnerCoRoutines(agent1, agent2, gameParams)
    val forwardModel = runner.runGame()
    println("Game over! Final state: $forwardModel")

}

class RemoteAgent(
    private val className: String,
    private val port: Int = 8080,
    private val logger: JsonLogger = JsonLogger(), // default: ignore = true
    private val debug: Boolean = false, 
) : PlanetWarsPlayer() {

    private val serverUrl: String = "ws://localhost:$port/ws"
    private lateinit var session: DefaultClientWebSocketSession
    private lateinit var client: HttpClient
    private lateinit var objectId: String

    private fun ensureConnected() {
        if (!::client.isInitialized) {
            client = HttpClient(CIO) { install(WebSockets) }
        }
    }

    override fun prepareToPlayAs(player: Player, params: GameParams, opponent: String?): String {
        runBlocking {
            ensureConnected()
            client.webSocket(serverUrl) {
                session = this
                objectId = initAgent(className)
                invokeRemoteMethod(
                    objectId,
                    method = "prepareToPlayAs",
                    args = listOf(player, params, opponent ?: "Anonymous"),
                    logger = logger,
                )
            }
        }
        return getAgentType()
    }

    override fun getAction(gameState: GameState): Action {
        return runBlocking {
            ensureConnected()
            var action: Action = Action.doNothing()
            client.webSocket(serverUrl) {
                session = this
                val response = invokeRemoteMethod(objectId, "getAction", args = listOf(gameState), logger = logger)
                val jsonResp = json.parseToJsonElement(response).jsonObject
//                print("Received response: $response\n")
                val result = jsonResp["result"]
                if (result != null && result is JsonObject) {
                    action = json.decodeFromJsonElement(Action.serializer(), result)
                    if (debug) println("Action received: $action")
                }
            }
            action
        }
    }

//    override fun getAgentType(): String = "Remote[$className]"

    override fun getAgentType(): String = runBlocking {
        ensureConnected()
        if (!::objectId.isInitialized) {
            // call prepare to play as
            prepareToPlayAs(Player.Player1, GameParams(), opponent = "RemoteAgent")
        }
        println("In Remote Agent, object id: $objectId")
        var agentType = "Remote[$className]" // fallback
        client.webSocket(serverUrl) {
            session = this
            val response = invokeRemoteMethod(objectId, "getAgentType", args = emptyList(), logger = logger)
            val jsonResp = json.parseToJsonElement(response).jsonObject
            val result = jsonResp["result"]
            if (result != null && result.toString().isNotBlank()) {
                agentType = result.toString().trim('"')  // remove surrounding quotes
            }
        }
        "$agentType (Remote)"
    }

    override fun processGameOver(finalState: GameState) {
        runBlocking {
            ensureConnected()
            client.webSocket(serverUrl) {
                session = this
                invokeRemoteMethod(
                    objectId = objectId,
                    method = "processGameOver",
                    args = listOf(finalState),
                    logger = logger,
                )
                endAgent(objectId)
            }
            client.close()
        }
    }
}
