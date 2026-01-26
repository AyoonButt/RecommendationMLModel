package com.api.postgres.utils

import io.jsonwebtoken.Jwts
import io.jsonwebtoken.SignatureAlgorithm
import io.jsonwebtoken.security.Keys
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Component
import java.util.*

@Component
class ServiceTokenGenerator {
    
    @Value("\${jwt.secret:your-secret-key-here}")
    private lateinit var jwtSecret: String
    
    fun generateServiceToken(): String {
        val key = Keys.hmacShaKeyFor(jwtSecret.toByteArray())
        val now = Date()
        val expiryDate = Date(now.time + 365L * 24 * 60 * 60 * 1000) // 1 year expiry
        
        return Jwts.builder()
            .setSubject("ml-service")
            .claim("userId", -1) // Special service user ID
            .claim("authorities", listOf("ROLE_SERVICE"))
            .claim("tokenType", "access")
            .setIssuedAt(now)
            .setExpiration(expiryDate)
            .signWith(key, SignatureAlgorithm.HS512)
            .compact()
    }
}

// Temporary controller to generate the token - ADD THIS TO YOUR SPRING API
@RestController
@RequestMapping("/dev")
class DevTokenController(private val tokenGenerator: ServiceTokenGenerator) {
    
    @GetMapping("/generate-service-token")
    fun generateServiceToken(): Map<String, String> {
        val token = tokenGenerator.generateServiceToken()
        return mapOf(
            "token" to token,
            "usage" to "Set this as SERVICE_AUTH_TOKEN in your ML service environment"
        )
    }
}