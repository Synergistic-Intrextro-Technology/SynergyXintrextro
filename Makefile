# Makefile
.PHONY: run build up down bench curl-status curl-route
run:
	uvicorn app.main:app --reload --port 8080
build:
	docker build -t cognitive-os:local .
up:
	docker compose up --build -d
down:
	docker compose down
curl-status:
	curl -s http://localhost:8080/status | jq .
curl-route:
	curl -s -X POST http://localhost:8080/route 	 -H "Content-Type: application/json" 	 -d '{"task":"qa","modality":"text","sla_ms":2000,"hints":{"mode":"hybrid"},"payload":"Explain the synergy loop."}' | jq .
