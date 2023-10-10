package hopsy;

import com.mongodb.*;
import com.mongodb.client.MongoCollection;
import org.bson.Document;
import static com.mongodb.client.model.Filters.eq;

class DBUtils {
  private static MongoClient usrMC =
      new MongoClient(
          new MongoClientURI(
              "mongodb+srv://haha: AKCpBjK6r3T3q85U6dXQ6IgYjK1Jut0I0MwOcBxtKjT6TqJ3Uzru @cluster0-gxgoh.mongodb.net/userdb?retryWrites=true&w=majority"));;

  DBUtils() {}

  static MongoClient getusrMC() {
    return usrMC;
  }

  static boolean insertDoc(MongoCollection<Document> collection, Document doc) { //insert a doc into the db
    try {
      collection.insertOne(doc);
    } catch (MongoException ex) {
      System.out.print("THERE WAS AN ERROR: " + ex.toString());
      return false;
    }
    return true;
  }

  static Document findDoc(MongoCollection<Document> collection, String key, String value) {
    return collection.find(eq(key, value)).first();
  }
}
