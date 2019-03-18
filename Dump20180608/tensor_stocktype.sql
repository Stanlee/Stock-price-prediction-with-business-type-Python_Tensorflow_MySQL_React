-- MySQL dump 10.13  Distrib 5.7.17, for Win64 (x86_64)
--
-- Host: localhost    Database: tensor
-- ------------------------------------------------------
-- Server version	5.6.40

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `stocktype`
--

DROP TABLE IF EXISTS `stocktype`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `stocktype` (
  `typecode` varchar(10) DEFAULT NULL,
  `typename` enum('의료정밀','종이목재','전기전자','화학','음식료품','섬유의복','기계','소형주','제조업','은행','중형주','종합','대형주','유통업','운수창고업','서비스업','의약품','증권','비금속광물','운수장비','건설업','금융업','철강금속','통신업','보험업','전기가스업') NOT NULL,
  PRIMARY KEY (`typename`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `stocktype`
--

LOCK TABLES `stocktype` WRITE;
/*!40000 ALTER TABLE `stocktype` DISABLE KEYS */;
INSERT INTO `stocktype` VALUES ('kospi01','의료정밀'),('kospi02','종이목재'),('kospi03','전기전자'),('kospi04','화학'),('kospi05','음식료품'),('kospi06','섬유의복'),('kospi07','기계'),('kospi08','소형주'),('kospi09','제조업'),('kospi10','은행'),('kospi11','중형주'),('kospi12','종합'),('kospi13','대형주'),('kospi14','유통업'),('kospi15','운수창고업'),('kospi16','서비스업'),('kospi17','의약품'),('kospi18','증권'),('kospi19','비금속광물'),('kospi20','운수장비'),('kospi21','건설업'),('kospi22','금융업'),('kospi23','철강금속'),('kospi24','통신업'),('kospi25','보험업'),('kospi26','전기가스업');
/*!40000 ALTER TABLE `stocktype` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2018-06-08 19:36:24
